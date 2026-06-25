import re
from typing import List, Dict, Tuple

class Guardrails:
    @staticmethod
    def detect_prompt_injection(query: str) -> bool:
        """
        Input Guardrail đơn giản: Chặn các câu lệnh cố tình ép hệ thống quên hướng dẫn (jailbreak).
        Thực tế cần dùng model LLM nhỏ để classify, ở đây dùng Regex cơ bản.
        """
        dangerous_patterns = [
            r"ignore previous instructions",
            r"bỏ qua các hướng dẫn",
            r"forget all",
            r"quên tất cả",
            r"system prompt"
        ]
        query_lower = query.lower()
        for pattern in dangerous_patterns:
            if re.search(pattern, query_lower):
                return True
        return False

    @staticmethod
    def extract_and_format_articles(llm_answer: str, retrieved_docs: List[Dict]) -> Tuple[List[str], List[str]]:
        """
        Output Guardrail: Hậu xử lý kết quả
        - Bắt buộc trích xuất các mẫu "Điều X" từ câu trả lời.
        - Đối chiếu với retrieved_docs để lấy law_id và law_name chuẩn.
        - Trả về relevant_docs và relevant_articles theo đúng format của BTC.
        """
        relevant_docs = set()
        relevant_articles = set()
        
        # Regex tìm "Điều X" hoặc "điều X" trong câu trả lời
        article_matches = re.finditer(r"(?i)(điều\s+\d+)", llm_answer)
        found_articles_in_text = [match.group(1).capitalize() for match in article_matches] # Chuẩn hóa "Điều X"
        
        # Đối chiếu với metadata của các văn bản đã truy hồi
        for doc in retrieved_docs:
            law_id = doc.metadata.get('law_id', '')
            law_name = doc.metadata.get('law_name', '')
            article_id = doc.metadata.get('article_id', '') # Vd: Điều 5
            
            # Formatting
            doc_str = f"{law_id}|{law_name}"
            article_str = f"{law_id}|{law_name}|{article_id}"
            
            # Thêm vào danh sách liên quan nếu Điều này có xuất hiện trong câu trả lời (hoặc nằm trong Top K truy hồi được coi là relevant luôn)
            # Tùy luật thi: nếu phải xuất hiện trong answer mới tính thì kiểm tra:
            if article_id in found_articles_in_text or not found_articles_in_text: 
                # Nếu LLM trả lời "Điều 5" và truy hồi cũng có "Điều 5", add vào list.
                # (Logic thực tế có thể mở rộng add toàn bộ top K nếu context phù hợp)
                relevant_docs.add(doc_str)
                relevant_articles.add(article_str)
                
        return list(relevant_docs), list(relevant_articles)
