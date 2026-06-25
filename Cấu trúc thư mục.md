r2ai/
├── app/
│   ├── server.py              # FastAPI server (quản lý endpoints API và serve frontend tĩnh)
│   └── static/                # Giao diện người dùng tĩnh (HTML, CSS, JS)
├── helpers/
│   ├── __init__.py
│   ├── config.py              # Quản lý cấu hình và biến môi trường (.env)
│   ├── constants.py           # Quản lý các system/user prompts và prompt guardrails templates
│   ├── decorators.py          # Bộ decorator tự động retry khi gặp lỗi kết nối API
│   ├── guardrails.py          # Bộ điều hướng an toàn (Prompt, Grounding, StreamLoop, Language)
│   ├── models.py              # Client tùy chỉnh cho Qdrant DB và Custom OpenAI (LLM, Reranker)
│   ├── pipeline.py            # RAG Pipeline cốt lõi kết hợp retrieval, reranking, capping và LLM
│   └── text_processing.py     # Hỗ trợ tách từ tiếng Việt (Word Segmentation) bằng underthesea
├── data/                      # Lưu trữ bộ Tokenizer và dữ liệu bổ trợ
├── Dockerfile                 # Dockerfile build image cho ứng dụng FastAPI
├── docker-compose.yml         # Cấu hình container chạy FastAPI và Qdrant DB song song
├── requirements.txt           # Danh sách các thư viện Python phụ thuộc
└── README.md                  # Tài liệu hướng dẫn sử dụng dự án

# Tính năng nổi bật

Tìm kiếm lai (Hybrid Search): Kết hợp giữa Dense Search (truy vấn ngữ nghĩa sâu) và Sparse Search (truy vấn từ khóa/BM25 thông qua thư viện FastEmbed) trên cơ sở dữ liệu vector Qdrant.
Đánh giá lại (Reranking): Sử dụng mô hình Reranker (như BGE-Reranker) thông qua API OpenAI tùy chỉnh để nâng cao độ chính xác của các tài liệu tìm thấy.
Bộ Guardrails toàn diện:
Prompt Guardrail: Sử dụng Regex và LLM phân tích ngữ nghĩa sâu để chặn đứng các cuộc tấn công Prompt Injection, Jailbreak, và rò rỉ prompt.
Grounding Guardrail: Tự động phân tích các dòng căn cứ pháp lý trong câu trả lời của LLM và đối chiếu chặt chẽ 100% với các chunks tài liệu gốc được truy vấn.
Stream Loop Guardrail: Theo dõi và tự động cắt luồng stream khi phát hiện vòng lặp vô hạn của token.
Language Guardrail: Áp dụng logit_bias để lọc bỏ hoàn toàn các ký tự ngoại lai (như CJK - tiếng Trung/Nhật/Hàn) trong lúc sinh từ đối với mô hình Gemma.
Hỗ trợ Streaming: Giao tiếp thời gian thực qua Server-Sent Events (SSE).
