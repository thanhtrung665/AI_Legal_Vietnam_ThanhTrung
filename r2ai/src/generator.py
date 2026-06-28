import json
from typing import List, Tuple
from pydantic import BaseModel, Field
from llama_index.core.schema import NodeWithScore
from llama_index.core.llms import ChatMessage, MessageRole

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    import torch
    from transformers import BitsAndBytesConfig
    from llama_index.llms.huggingface import HuggingFaceLLM
except ImportError:
    raise ImportError("Vui lòng cài đặt: pip install python-dotenv torch transformers bitsandbytes accelerate llama-index-llms-huggingface")

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR.parent / ".env")

class LegalAnswer(BaseModel):
    answer: str = Field(description="Câu trả lời sinh ra từ LLM")
    relevant_docs: List[str] = Field(description="Danh sách văn bản liên quan (Rule-based)")
    relevant_articles: List[str] = Field(description="Danh sách điều luật liên quan (Rule-based)")

# [TỐI ƯU 2]: Nâng cấp System Prompt
SYSTEM_PROMPT = """Bạn là một chuyên gia pháp lý và Luật sư AI xuất sắc tại Việt Nam.
Nhiệm vụ của bạn là tư vấn và trả lời câu hỏi pháp lý MỘT CÁCH CHÍNH XÁC, NGẮN GỌN DỰA TRÊN CÁC TRÍCH ĐOẠN VĂN BẢN (Context) được cung cấp.

QUY TẮC BẮT BUỘC (CRITICAL RULES):
1. ĐỐI VỚI CÂU HỎI XÁC NHẬN (Ví dụ: "Có được không?", "Phải không?", "Đúng không?"): 
   - Bạn BẮT BUỘC phải mở đầu câu trả lời bằng "Có." hoặc "Không." hoặc "Có, nhưng tùy trường hợp.". 
   - Sau đó mới giải thích chi tiết dựa vào Context.
2. CÁCH TRÌNH BÀY: 
   - LUÔN LUÔN trích dẫn tên Điều luật cụ thể khi giải thích (Ví dụ: "Theo Điều X của Luật Y...").
   - Trả lời súc tích, đi thẳng vào trọng tâm, sử dụng bullet points nếu cần liệt kê.
3. TÍNH CHÍNH XÁC: 
   - Tuyệt đối không bịa đặt (hallucinate) thông tin ngoài Context. 
   - Nếu Context không chứa câu trả lời, chỉ cần đáp: "Dựa trên dữ liệu pháp lý hiện tại, không có đủ thông tin để trả lời."
"""

class LegalGenerator:
    def __init__(self, model: str = "Qwen/Qwen2.5-7B-Instruct"):
        print(f"[INFO] Khởi tạo LLM HuggingFace (Model: {model}, 4-bit Quantization)")
        
        hf_token = os.getenv("HUGGINGFACE_TOKEN_API")
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token
            
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        
        self.llm = HuggingFaceLLM(
            model_name=model,
            tokenizer_name=model,
            model_kwargs={"quantization_config": quantization_config},
            generate_kwargs={"temperature": 0.01, "do_sample": False},
            max_new_tokens=1024,
            device_map="auto"
        )
        
    def _extract_references(self, nodes: List[NodeWithScore]) -> Tuple[List[str], List[str]]:
        relevant_docs = set()
        relevant_articles = set()
        
        for n in nodes:
            ma_vb = n.node.metadata.get("mã_văn_bản", "").strip()
            ten_vb = n.node.metadata.get("tên_văn_bản", "").strip()
            dieu = n.node.metadata.get("điều", "").strip()
            
            if ma_vb and ten_vb:
                doc_str = f"{ma_vb}|{ten_vb}"
                relevant_docs.add(doc_str)
                
                if dieu:
                    article_str = f"{ma_vb}|{ten_vb}|{dieu}"
                    relevant_articles.add(article_str)
                    
        return sorted(list(relevant_docs)), sorted(list(relevant_articles))
        
    def generate_answer(self, query: str, retrieved_nodes: List[NodeWithScore]) -> LegalAnswer:
        # [TỐI ƯU 1]: Short-circuit nếu không có Context
        if not retrieved_nodes:
            return LegalAnswer(
                answer="Dựa trên dữ liệu pháp lý hiện tại, không có đủ thông tin để trả lời.",
                relevant_docs=[],
                relevant_articles=[]
            )

        docs, articles = self._extract_references(retrieved_nodes)
        
        context_str = ""
        for i, n in enumerate(retrieved_nodes, 1):
            ma_vb = n.node.metadata.get("mã_văn_bản", "Unknown")
            dieu = n.node.metadata.get("điều", "Unknown")
            context_str += f"--- Bắt đầu Nguồn {i} ({dieu} - {ma_vb}) ---\n{n.node.text}\n--- Kết thúc Nguồn {i} ---\n\n"
            
        user_msg_content = (
            f"NGỮ CẢNH PHÁP LÝ:\n{context_str}\n"
            f"CÂU HỎI CỦA NGƯỜI DÙNG:\n{query}\n\n"
            f"Hãy thực hiện vai trò Luật sư và trả lời câu hỏi dựa trên Ngữ cảnh pháp lý trên."
        )
        
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT),
            ChatMessage(role=MessageRole.USER, content=user_msg_content)
        ]
        
        try:
            response = self.llm.chat(messages)
            llm_answer = response.message.content.strip()
        except Exception as e:
            print(f"[ERROR LLM Generation]: {e}")
            llm_answer = "Xin lỗi, đã có lỗi kết nối hệ thống trong quá trình sinh câu trả lời."
            
        return LegalAnswer(
            answer=llm_answer,
            relevant_docs=docs,
            relevant_articles=articles
        )


if __name__ == "__main__":
    from llama_index.core.schema import TextNode
    
    print("="*50)
    print("BẮT ĐẦU UNIT TEST PHASE 4: LLM GENERATOR")
    print("="*50)
    
    # Tạo dữ liệu giả (Mock Data) đóng vai trò là Top 2 Chunks sau khi truy hồi
    mock_nodes = [
        NodeWithScore(
            node=TextNode(
                text="Doanh nghiệp siêu nhỏ, doanh nghiệp nhỏ và doanh nghiệp vừa được xác định theo lĩnh vực nông nghiệp, lâm nghiệp, thủy sản; công nghiệp và xây dựng; thương mại và dịch vụ.",
                metadata={
                    "mã_văn_bản": "Luật 04/2017/QH14",
                    "tên_văn_bản": "Luật hỗ trợ doanh nghiệp nhỏ và vừa",
                    "điều": "Điều 4. Tiêu chí xác định doanh nghiệp nhỏ và vừa"
                }
            ),
            score=0.95
        ),
        NodeWithScore(
            node=TextNode(
                text="Việc hỗ trợ doanh nghiệp nhỏ và vừa phải tôn trọng quy luật thị trường, phù hợp với điều ước quốc tế mà nước Cộng hòa xã hội chủ nghĩa Việt Nam là thành viên.",
                metadata={
                    "mã_văn_bản": "Luật 04/2017/QH14",
                    "tên_văn_bản": "Luật hỗ trợ doanh nghiệp nhỏ và vừa",
                    "điều": "Điều 5. Nguyên tắc hỗ trợ doanh nghiệp nhỏ và vừa"
                }
            ),
            score=0.85
        )
    ]
    
    query = "Nguyên tắc hỗ trợ doanh nghiệp nhỏ và vừa có cần tôn trọng quy luật thị trường không?"
    
    # Lưu ý: Unit Test này sẽ throw Error nếu bạn chưa bật vLLM local tại port 8000.
    # Tuy nhiên Code Rule-based Extraction vẫn sẽ chạy mượt mà.
    generator = LegalGenerator()
    
    print(f"\n[INFO] Câu hỏi: {query}")
    print("[INFO] Đang sinh câu trả lời...")
    
    final_output = generator.generate_answer(query, mock_nodes)
    
    print("\n[SUCCESS] Cấu trúc Output Pydantic:")
    # Chuyển model thành JSON để dễ đọc
    print(json.dumps(final_output.model_dump(), ensure_ascii=False, indent=2))
