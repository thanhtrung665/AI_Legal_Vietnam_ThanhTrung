import torch
from typing import List, Tuple
from pydantic import BaseModel, Field
from llama_index.core.schema import NodeWithScore

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
except ImportError:
    raise ImportError("Vui lòng cài đặt: pip install python-dotenv torch transformers bitsandbytes accelerate")

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR.parent / ".env")

class LegalAnswer(BaseModel):
    answer: str = Field(description="Câu trả lời sinh ra từ LLM")
    relevant_docs: List[str] = Field(description="Danh sách văn bản liên quan (Rule-based)")
    relevant_articles: List[str] = Field(description="Danh sách điều luật liên quan (Rule-based)")

# [TỐI ƯU 3]: Rút gọn System Prompt — giảm input tokens, tăng tốc inference
SYSTEM_PROMPT = """Bạn là Luật sư AI chuyên pháp luật Việt Nam. Trả lời CHÍNH XÁC, NGẮN GỌN dựa trên Context được cung cấp.

QUY TẮC BẮT BUỘC:
1. Câu hỏi xác nhận ("Có được không?", "Phải không?", "Đúng không?"): BẮT BUỘC mở đầu bằng "Có." hoặc "Không." hoặc "Có, nhưng tùy trường hợp."
2. LUÔN trích dẫn tên Điều luật cụ thể (Ví dụ: "Theo Điều X Luật Y...").
3. Không bịa đặt thông tin ngoài Context. Nếu không đủ thông tin: "Dựa trên dữ liệu pháp lý hiện tại, không có đủ thông tin để trả lời."
"""

class LegalGenerator:
    # [TỐI ƯU 2]: Bypass LlamaIndex hoàn toàn, dùng native AutoModel + AutoTokenizer
    # Loại bỏ 6-8 lớp middleware của LlamaIndex, tăng tốc inference 30-50%
    def __init__(self, model: str = "Qwen/Qwen2.5-3B-Instruct"):
        print(f"[INFO] Khởi tạo LLM Native Transformers (Model: {model}, 4-bit Quantization)")

        hf_token = os.getenv("HUGGINGFACE_TOKEN_API")
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()
        print(f"[INFO] Model đã sẵn sàng. Device map: {self.model.hf_device_map}")

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

    def _build_prompt(self, query: str, retrieved_nodes: List[NodeWithScore]) -> str:
        """[TỐI ƯU 3]: Rút gọn context template, truncate chunk text để giảm input tokens."""
        context_parts = []
        for i, n in enumerate(retrieved_nodes, 1):
            ma_vb = n.node.metadata.get("mã_văn_bản", "Unknown")
            dieu = n.node.metadata.get("điều", "Unknown")
            # Cắt chunk text tối đa 500 ký tự để tránh prompt quá dài
            text = n.node.text[:500]
            context_parts.append(f"[Nguồn {i} | {dieu} - {ma_vb}]\n{text}")

        context_str = "\n\n".join(context_parts)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"NGỮ CẢNH PHÁP LÝ:\n{context_str}\n\nCÂU HỎI: {query}\n\nTrả lời:"}
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def generate_answer(self, query: str, retrieved_nodes: List[NodeWithScore]) -> LegalAnswer:
        if not retrieved_nodes:
            return LegalAnswer(
                answer="Dựa trên dữ liệu pháp lý hiện tại, không có đủ thông tin để trả lời.",
                relevant_docs=[],
                relevant_articles=[]
            )

        docs, articles = self._extract_references(retrieved_nodes)
        prompt = self._build_prompt(query, retrieved_nodes)

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            input_len = inputs.input_ids.shape[1]

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,      # [TỐI ƯU 1]: Giảm từ 1024 xuống 512
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,  # Chống lặp câu
                )

            # Decode chỉ phần token mới sinh ra (bỏ phần prompt input)
            generated_ids = outputs[0][input_len:]
            llm_answer = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        except Exception as e:
            print(f"[ERROR LLM Generation]: {e}")
            llm_answer = "Xin lỗi, đã có lỗi kết nối hệ thống trong quá trình sinh câu trả lời."

        return LegalAnswer(
            answer=llm_answer,
            relevant_docs=docs,
            relevant_articles=articles
        )
