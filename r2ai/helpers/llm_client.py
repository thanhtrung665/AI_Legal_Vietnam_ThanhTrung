import os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from dotenv import load_dotenv

load_dotenv()

class LLMClient:
    def __init__(self, model_name="Qwen/Qwen2.5-7B-Instruct"):
        """
        Khởi tạo Model. Lưu ý: 7B model cần ~14GB VRAM. 
        Trong môi trường R&D, nếu không có GPU mạnh, hãy sử dụng Ollama hoặc Inference API.
        Ở đây dùng HuggingFace Transformers pipeline làm chuẩn.
        """
        print(f"Loading LLM {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Load model in 4-bit/8-bit if using bitsandbytes for lower VRAM, here keep it simple
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype="auto" # requires torch
        )
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=512,
            temperature=0.1, # Nhiệt độ thấp để giảm ảo giác (hallucination)
            top_p=0.9
        )

    def generate_answer(self, prompt: str) -> str:
        messages = [
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        outputs = self.pipe(text)
        return outputs[0]["generated_text"][len(text):]

# Dummy Client để test pipeline nếu không có đủ RAM
class DummyLLMClient:
    def generate_answer(self, prompt: str) -> str:
        return "Theo Điều 15 Luật Doanh nghiệp 2020, đây là câu trả lời dummy do máy không có GPU."
