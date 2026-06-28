import os
# [FIX OOM]: Cấu hình tối ưu chống phân mảnh VRAM trước khi import torch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import zipfile
from pathlib import Path
from tqdm import tqdm

from indexer import build_qdrant_index, init_settings, QDRANT_DATA_DIR
from retriever import LegalRetriever
from generator import LegalGenerator

BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent
INPUT_JSON = PROJECT_ROOT / "R2AIStage1DATA.json"
OUTPUT_JSON = PROJECT_ROOT / "results.json"
OUTPUT_ZIP = PROJECT_ROOT / "submission.zip"

def process_question(retriever: LegalRetriever, generator: LegalGenerator, q_id: int, question: str) -> dict:
    try:
        import torch
        with torch.no_grad():
            retrieved_nodes = retriever.retrieve(question)
            legal_answer = generator.generate_answer(question, retrieved_nodes)
        
        return {
            "id": q_id,
            "question": question,
            "answer": legal_answer.answer,
            "relevant_docs": legal_answer.relevant_docs,
            "relevant_articles": legal_answer.relevant_articles
        }
    except Exception as e:
        # [TỐI ƯU]: Dùng tqdm.write thay cho print để không làm vỡ thanh tiến trình
        tqdm.write(f"[ERROR] Lỗi xử lý ID {q_id}: {e}")
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
        # Ghi duy nhất file json, không tạo thư mục
        zipf.write(json_path, arcname=json_path.name)
    print(f"[SUCCESS] Đóng gói thành công! Sẵn sàng nộp bài: {zip_path.name}")

def main():
    print("="*50)
    print("BẮT ĐẦU PHASE 5: EVALUATION & PIPELINE (1591 CÂU)")
    print("="*50)
    
    init_settings()
    
    if not os.path.exists(QDRANT_DATA_DIR):
        print(f"[WARNING] Không tìm thấy DB tại {QDRANT_DATA_DIR}. Vui lòng chạy Phase 1&2 trước!")
        return # Dừng luôn nếu chưa có Data
        
    # LƯU Ý: Phải đảm bảo collection_name ở đây khớp với lúc bạn chạy Indexer (Phase 2)
    # Ví dụ ở Phase 2 bạn tạo "legal_vn" thì ở đây phải là "legal_vn"
    index = build_qdrant_index(nodes=[], collection_name="legal_vn_test")
    
    retriever = LegalRetriever(index=index, retrieve_top_k=10, rerank_top_k=5)
    generator = LegalGenerator()
    
    if not INPUT_JSON.exists():
        raise FileNotFoundError(f"Không tìm thấy file {INPUT_JSON}")
        
    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        questions_data = json.load(f)
        
    results = []
    
    for q_item in tqdm(questions_data, desc="[Tiến độ AI Legal]", unit="câu"):
        q_id = q_item.get("id")
        question = q_item.get("question", "")
        
        if not question:
            continue
            
        result = process_question(retriever, generator, q_id, question)
        results.append(result)
        
        # [FIX OOM]: Dọn rác VRAM sau mỗi câu hỏi để tránh rò rỉ bộ nhớ
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        if len(results) % 100 == 0:
            with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
                
    print(f"\n[INFO] Lưu toàn bộ kết quả ra {OUTPUT_JSON.name}...")
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
        
    zip_results(OUTPUT_JSON, OUTPUT_ZIP)
    
    print("\n" + "="*50)
    print("🏆 HOÀN TẤT DỰ ÁN RAG PHÁP LUẬT VIỆT NAM!")
    print(f"👉 File nộp bài của bạn nằm tại: {OUTPUT_ZIP.absolute()}")
    print("Mang đi nộp ngay và chờ xem mình lọt top mấy Leaderboard nhé!")
    print("="*50)

if __name__ == "__main__":
    main()