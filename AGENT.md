# Quy tắc

**Quy tắc Mô hình (Model Constraints):** Chỉ được phép đề xuất, sử dụng hoặc viết code API gọi các mô hình LLM có tham số dưới 14B và được phát hành trước ngày 01/03/2026. Không được tự ý gọi các model khép kín khổng lồ như GPT-4 hay Claude 3.5 Sonnet trong pipeline chính thức.
**Mục tiêu tái lập:** Mọi code tải mô hình (Hugging Face, vLLM, Ollama) đều phải rõ ràng để Ban tổ chức có thể tái lập kết quả.
**Zero-Hallucination (Không bịa đặt):** Agent phải thiết kế Prompt cho LLM sao cho LLM chỉ được phép trả lời dựa trên nội dung các Điều luật được truy hồi (retrieved docs). Nếu thông tin không có trong text truy hồi, LLM phải trả lời là "Không đủ thông tin".
**Chunking theo "Điều":** Bắt buộc hệ thống chia nhỏ văn bản (Chunking) phải giữ nguyên vẹn nội dung ở cấp độ "Điều", không được cắt ngang một Điều luật.
**Cấu trúc JSON tuyệt đối:** Đầu ra cuối cùng của pipeline code phải là một file tên chính xác là results.json.
**Quy tắc trích xuất Điều luật:** Agent phải viết một module Hậu xử lý (Post-processing) bằng Regex hoặc Pattern Matching để bắt buộc trích xuất chuỗi định dạng "Điều X" từ câu trả lời của LLM.
**Định dạng mảng (Array Format):**

* relevant_docs bắt buộc phải theo format: <mã văn bản>|<tên văn bản>.
* relevant_articles bắt buộc phải theo format: <mã văn bản>|<tên văn bản>|<điều>.
Lưu ý cho Agent: "Điều X" phải được chuẩn hóa đúng chữ "Điều" viết hoa chữ Đ, khoảng trắng và số X.
**Tối ưu hóa F2 Score:** Yêu cầu Agent thiết kế logic truy hồi ưu tiên độ bao phủ (Recall), vì điểm đánh giá truy hồi sử dụng chỉ số $F_2$ (đặt nặng Recall hơn Precision). Gợi ý Agent sử dụng Hybrid Search (BM25 + Vector).
**Bảo mật:** tạo file .env để lưu trữ API, Token, password và những thông tin cần bảo mật
