import os
# [FIX OOM]: Cấu hình tối ưu chống phân mảnh VRAM trước khi import torch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import zipfile
import torch
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from indexer import build_qdrant_index, init_settings, QDRANT_DATA_DIR
from retriever import LegalRetriever
from generator import LegalGenerator

BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent
INPUT_JSON = PROJECT_ROOT / "R2AIStage1DATA.json"
OUTPUT_JSON = PROJECT_ROOT / "results.json"
OUTPUT_ZIP = PROJECT_ROOT / "submission.zip"

CHECKPOINT_INTERVAL = 50    # Lưu checkpoint mỗi 50 câu (tránh mất dữ liệu nếu crash)
RETRIEVAL_WORKERS = 4       # Số luồng song song để query Qdrant (I/O bound)

# ─────────────────────────────────────────────────
# [TỐI ƯU RESUME]: Load kết quả đã làm, skip câu đã xử lý
# ─────────────────────────────────────────────────
def load_existing_results(output_path: Path) -> dict:
    """Load kết quả cũ từ file JSON để có thể tiếp tục từ điểm đã dừng."""
    if not output_path.exists():
        return {}
    try:
        with open(output_path, 'r', encoding='utf-8') as f:
            existing = json.load(f)
        done_map = {item["id"]: item for item in existing if "id" in item}
        print(f"[RESUME] ✅ Tìm thấy {len(done_map)} câu đã xử lý. Tiếp tục từ câu chưa làm...")
        return done_map
    except Exception as e:
        print(f"[RESUME] ⚠️ Không đọc được file cũ ({e}). Bắt đầu lại từ đầu.")
        return {}

# ─────────────────────────────────────────────────
# [TỐI ƯU BATCH RETRIEVAL]: Retrieve nhiều câu song song
# ─────────────────────────────────────────────────
def batch_retrieve(retriever: LegalRetriever, batch: list) -> list:
    """Retrieve song song nhiều câu hỏi cùng lúc bằng ThreadPoolExecutor (I/O bound)."""
    results_map = {}

    def _retrieve_one(item):
        q_id = item["id"]
        question = item["question"]
        try:
            nodes = retriever.retrieve(question)
            return q_id, nodes
        except Exception as e:
            tqdm.write(f"[ERROR Retrieval] ID {q_id}: {e}")
            return q_id, []

    with ThreadPoolExecutor(max_workers=RETRIEVAL_WORKERS) as executor:
        futures = {executor.submit(_retrieve_one, item): item for item in batch}
        for future in as_completed(futures):
            q_id, nodes = future.result()
            results_map[q_id] = nodes

    return results_map

def process_question(generator: LegalGenerator, q_id: int, question: str, retrieved_nodes) -> dict:
    """Chỉ thực hiện bước Generate (LLM) với kết quả retrieval đã có sẵn."""
    try:
        legal_answer = generator.generate_answer(question, retrieved_nodes)
        return {
            "id": q_id,
            "question": question,
            "answer": legal_answer.answer,
            "relevant_docs": legal_answer.relevant_docs,
            "relevant_articles": legal_answer.relevant_articles
        }
    except Exception as e:
        tqdm.write(f"[ERROR Generate] ID {q_id}: {e}")
        return {
            "id": q_id,
            "question": question,
            "answer": "Dựa trên dữ liệu pháp lý hiện tại, không có đủ thông tin để trả lời.",
            "relevant_docs": [],
            "relevant_articles": []
        }

def zip_results(json_path: Path, zip_path: Path):
    print(f"[INFO] Đang nén file kết quả thành {zip_path.name}...")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(json_path, arcname=json_path.name)
    print(f"[SUCCESS] Đóng gói thành công! Sẵn sàng nộp bài: {zip_path.name}")

def main():
    print("="*55)
    print("BẮT ĐẦU PHASE 5: EVALUATION & PIPELINE (2000 CÂU)")
    print("   [OPTIMIZED]: Resume + Batch Retrieval + Native LLM")
    print("="*55)

    init_settings()

    if not os.path.exists(QDRANT_DATA_DIR):
        print(f"[WARNING] Không tìm thấy DB tại {QDRANT_DATA_DIR}. Vui lòng chạy Phase 1&2 trước!")
        return

    index = build_qdrant_index(nodes=[], collection_name="legal_vn_test")

    retriever = LegalRetriever(index=index, retrieve_top_k=10, rerank_top_k=5)
    generator = LegalGenerator()

    if not INPUT_JSON.exists():
        raise FileNotFoundError(f"Không tìm thấy file {INPUT_JSON}")

    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        questions_data = json.load(f)

    # [TỐI ƯU RESUME]: Đọc kết quả cũ, chỉ xử lý những câu chưa có
    done_map = load_existing_results(OUTPUT_JSON)
    results = list(done_map.values())

    pending = [
        q for q in questions_data
        if q.get("id") not in done_map and q.get("question", "").strip()
    ]
    print(f"[INFO] Tổng cộng: {len(questions_data)} câu | Đã xử lý: {len(done_map)} | Còn lại: {len(pending)} câu")

    # ─────────────────────────────────────────────
    # PIPELINE: Retrieve theo batch (song song) → Generate tuần tự (GPU)
    # ─────────────────────────────────────────────
    batch_size = RETRIEVAL_WORKERS * 2  # Mỗi lần pre-fetch gấp đôi số worker

    with tqdm(total=len(pending), desc="[Tiến độ AI Legal]", unit="câu") as pbar:
        for batch_start in range(0, len(pending), batch_size):
            batch = pending[batch_start: batch_start + batch_size]

            # BƯỚC 1: Retrieve tất cả câu trong batch song song (không bị GPU block)
            retrieved_map = batch_retrieve(retriever, batch)

            # BƯỚC 2: Generate từng câu tuần tự trên GPU
            for q_item in batch:
                q_id = q_item["id"]
                question = q_item["question"]
                nodes = retrieved_map.get(q_id, [])

                result = process_question(generator, q_id, question, nodes)
                results.append(result)
                pbar.update(1)

                # [TỐI ƯU]: Chỉ lưu checkpoint mỗi CHECKPOINT_INTERVAL câu
                if len(results) % CHECKPOINT_INTERVAL == 0:
                    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
                        json.dump(results, f, ensure_ascii=False, indent=2)
                    tqdm.write(f"[CHECKPOINT] 💾 Đã lưu {len(results)}/{len(questions_data)} kết quả")

    # [TỐI ƯU OOM]: Chỉ gọi empty_cache SAU KHI hoàn tất, không phải mỗi câu
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n[INFO] Lưu toàn bộ kết quả ra {OUTPUT_JSON.name}...")
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    zip_results(OUTPUT_JSON, OUTPUT_ZIP)

    print("\n" + "="*55)
    print("🏆 HOÀN TẤT DỰ ÁN RAG PHÁP LUẬT VIỆT NAM!")
    print(f"👉 File nộp bài của bạn nằm tại: {OUTPUT_ZIP.absolute()}")
    print("Mang đi nộp ngay và chờ xem mình lọt top mấy Leaderboard nhé!")
    print("="*55)

if __name__ == "__main__":
    main()