import sys
import os
import json
import time
from pathlib import Path

# Fix Unicode encode error trên Windows Console
sys.stdout.reconfigure(encoding='utf-8')

# Thêm đường dẫn gốc r2ai vào sys path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from helpers.models import RetrievalModels
from helpers.pipeline import LegalRAGPipeline
from helpers.constants import SYSTEM_PROMPT
from helpers.llm_client import DummyLLMClient # Đổi thành LLMClient nếu chạy GPU
from helpers.guardrails import Guardrails

def retry_with_backoff(fn, max_retries=3, initial_delay=5):
    """Retry một hàm với exponential backoff khi gặp lỗi timeout"""
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            delay = initial_delay * (2 ** attempt)
            print(f"  [RETRY] Lỗi kết nối (lần {attempt+1}/{max_retries}), đợi {delay}s rồi thử lại...")
            time.sleep(delay)

def process_questions(input_file: str, output_file: str):
    print(f"Loading data from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Kiểm tra xem có file kết quả trung gian không (để resume)
    partial_file = output_file + ".partial"
    results = []
    start_idx = 0
    
    if os.path.exists(partial_file):
        with open(partial_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        start_idx = len(results)
        print(f"Tìm thấy {start_idx} kết quả đã xử lý trước đó. Tiếp tục từ câu {start_idx + 1}...")
        
    # Khởi tạo Pipeline
    models = RetrievalModels()
    pipeline = LegalRAGPipeline(models)
    
    # Khởi tạo LLM
    # Trong môi trường thực tế, dùng LLMClient thay cho DummyLLMClient
    llm = DummyLLMClient() 
    
    print(f"Bắt đầu xử lý {len(data)} câu hỏi (từ câu {start_idx + 1})...")
    for idx in range(start_idx, len(data)):
        item = data[idx]
        q_id = item.get('id')
        question = item.get('question')
        
        print(f"Processing Q{q_id} ({idx+1}/{len(data)}): {question[:50]}...")
        
        # 1. Guardrail Input
        if Guardrails.detect_prompt_injection(question):
            print("  Phát hiện Prompt Injection!")
            answer = "Yêu cầu của bạn vi phạm chính sách an toàn."
            rel_docs, rel_arts = [], []
        else:
            # 2. Retrieval & Reranking (có retry)
            top_docs = retry_with_backoff(
                lambda q=question: pipeline.retrieve(q, top_k_retrieval=15, top_k_rerank=5),
                max_retries=3,
                initial_delay=5
            )
            
            # Format context cho LLM
            context_str = "\n".join([f"- {doc.page_content}" for doc in top_docs])
            prompt = SYSTEM_PROMPT.format(context=context_str, question=question)
            
            # 3. LLM Generation
            answer = llm.generate_answer(prompt)
            
            # 4. Guardrail Output & Formatting
            rel_docs, rel_arts = Guardrails.extract_and_format_articles(answer, top_docs)
            
        # Thêm vào mảng kết quả
        results.append({
            "id": q_id,
            "question": question,
            "answer": answer,
            "relevant_docs": rel_docs,
            "relevant_articles": rel_arts
        })
        
        # Lưu kết quả trung gian mỗi 20 câu
        if (idx + 1) % 20 == 0:
            print(f"  Đã xử lý {idx + 1}/{len(data)}. Lưu kết quả trung gian...")
            with open(partial_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
            
    # Lưu file results.json cuối cùng
    print(f"Saving results to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    # Xóa file trung gian nếu hoàn tất
    if os.path.exists(partial_file):
        os.remove(partial_file)
        
    print("Done! Bạn có thể nén file results.json thành submission.zip để nộp bài.")

if __name__ == "__main__":
    # Đường dẫn file input giả định (hoặc truyền vào tùy ý)
    input_path = os.path.join(str(Path(__file__).resolve().parent.parent.parent), "R2AIStage1DATA.json")
    output_path = os.path.join(str(Path(__file__).resolve().parent.parent.parent), "results.json")
    
    # Nếu file input tồn tại, tiến hành chạy
    if os.path.exists(input_path):
        process_questions(input_path, output_path)
    else:
        print(f"Không tìm thấy file data test tại {input_path}")
