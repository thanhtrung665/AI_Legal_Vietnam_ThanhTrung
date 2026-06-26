# Tổng quan về cuộc thi Road to AI

Hãy làm việc với vai trò một AI Engineer và Quản lý dự án AI có hơn 15 năm kinh nghiệm, kiến thức sâu rộng, chuyên nghiệp về LLM, RAG, Chatbot trong lĩnh vực Luật Việt Nam. Nhiệm vụ của bạn sẽ xây một hệ thống Chatbot AI Legal luật Việt Nam để tôi tham gia cuộc thi Road to AI chuẩn chỉnh, chuyên nghiệp và an toàn, hiệu suất cao, chính xác cao. Kết quả mong muốn của tôi là sản phẩm AI Chatbot có thể truy xuất, trả lời các câu hỏi liên quan pháp lý chính xác, có cấu trúc tốt và đạt tất cả yêu cầu của cuộc thi Road to AI đề ra. Bên cạnh đó trả lời chính xác hơn 90% các câu hỏi từ file R2AIStage1DATA.json để nộp bài lên leaderboard nhé. Hãy cùng nhau làm ra 1 sản phẩm tốt nhất, chất lượng nhất nhé!!!

- Về thông tin cuộc thi cũng như các yêu cầu, chỉ số đánh giá, quy tắc nộp bài bạn có thể xem ở file : C:\Users\asus\Downloads\AI_Legal\Tổng quan cuộc thi.md hoặc trang web này: <https://leaderboard.aiguru.com.vn/competitions/13/>
- Về định hướng hệ thống cơ bản bạn xem ở file : C:\Users\asus\Downloads\AI_Legal\Định hướng hệ thống.md
- Về nguồn dữ liệu thì thông tin ở file: C:\Users\asus\Downloads\AI_Legal\Nguồn dữ liệu.md
- Về cấu trúc thư mục và code demo dự kiến ban tổ chức cung cấp bạn xem ở thư mục: C:\Users\asus\Downloads\AI_Legal\Cấu trúc thư mục.md và <https://github.com/AI-Guru-R2AI/R2AI-MENTOR-DAY3>
- Các kiến thức liên quan mà tôi được training trong cuộc thi bạn xem ở các file PDF sau: C:\Users\asus\Downloads\AI_Legal\Kỹ thuật truy vấn.pdf, C:\Users\asus\Downloads\AI_Legal\Triển khai hệ thống lên Production.pdf,
- File câu hỏi để làm kiểm tra hệ thống ban tổ chức cung cấp là file Json này: C:\Users\asus\Downloads\AI_Legal\R2AIStage1DATA.json
- Ngoài ra tôi có lưu ý một xíu quy tắc ở file : C:\Users\asus\Downloads\AI_Legal\AGENT.md, bạn có thể sửa, bổ sung hoặc cải thiện trong quá trình thực hiện xây dựng hệ thống nhé.
TỔNG QUAN TECH STACK (CÔNG NGHỆ KHUYÊN DÙNG)
Orchestration Framework: LlamaIndex (Vượt trội hơn LangChain trong việc xử lý Hierarchical Data và Parent-Child Retrieval).

Data Parsing: PyMuPDF (pdf), python-docx (docx), regex (nhận diện cấu trúc luật).

Vector Database: Qdrant (Hỗ trợ Hybrid Search và Metadata Filtering cực tốt, dễ triển khai local qua Docker).

Embedding Model: BAAI/bge-m3 (Hỗ trợ đa ngôn ngữ, rất mạnh tiếng Việt, có cả dense vector và sparse vector cho từ khóa).

Reranker Model: BAAI/bge-reranker-m3 hoặc Cross-Encoder chuyên biệt.

LLM (Generation): Qwen2.5-7B/14B-Instruct (mã nguồn mở, tiếng Việt tốt, follow format JSON cực chuẩn) hoặc Gemini 1.5 Flash/Pro (nếu BTC cho phép dùng API).

Structured Output Enforcer: Pydantic kết hợp Instructor hoặc tính năng Function Calling mặc định của model.

⚙️ CHI TIẾT CÁC BƯỚC TRIỂN KHAI (PIPELINE)
Bước 1: Tiền xử lý & Nạp dữ liệu (Data Ingestion & Parsing)
Mục tiêu là biến file văn bản markdown thành các block văn bản có cấu trúc, gắn chặt với metadata_mapping.json.

Công cụ: Python, regex, json.

Thực thi:

Script đọc tên file , đối chiếu với metadata_mapping.json để lấy mã_văn_bản và tên_văn_bản.

Dùng Regex để bóc tách văn bản theo phân cấp. Cấu trúc Regex cơ bản để bắt các "Điều": ^Điều\s+\d+\..

Tạo ra các đối tượng JSON trung gian chứa nguyên văn của từng "Điều", "Khoản", "Điểm", kèm theo metadata tuyệt đối:

JSON
{
  "text": "1. Doanh nghiệp phải nộp hồ sơ trong 30 ngày...",
  "metadata": {
    "mã_văn_bản": "Luật 59/2020/QH14",
    "tên_văn_bản": "Luật Doanh nghiệp",
    "điều": "Điều 15"
  }
}
Bước 2: Phân mảnh dữ liệu (Hierarchical Chunking)
Công cụ: LlamaIndex (cụ thể là NodeParser).

Thực thi:

Sử dụng Parent-Child Chunking. Thiết lập "Điều" làm Parent Node. Thiết lập các "Khoản/Điểm" bên trong "Điều" đó làm Child Nodes.

Chỉ Embedding các Child Nodes để tăng độ nhạy khi tìm kiếm, nhưng cấu hình hệ thống luôn liên kết ngược về Parent Node để giữ toàn vẹn ngữ cảnh.

Bước 3: Embedding & Đưa vào Vector Database
Luật pháp đòi hỏi tính chính xác của từ khóa (ví dụ: "công ty TNHH 1 thành viên" khác với "2 thành viên"). Do đó, không chỉ dùng Semantic Search thông thường.

Công cụ: Qdrant, bge-m3.

Thực thi:

Sử dụng cơ chế Hybrid Search: Sinh ra Dense Vector (tìm kiếm ngữ nghĩa) và Sparse Vector / BM25 (tìm kiếm từ khóa chính xác).

Lưu toàn bộ vector và metadata vào Qdrant Collection.

Bước 4: Kỹ thuật Truy xuất (Advanced Retrieval) & Reranking
Công cụ: LlamaIndex Retriever, AutoMergingRetriever, BGE-Reranker.

Thực thi:

Query Rewrite (Tùy chọn): Dùng LLM nhỏ viết lại câu hỏi của user thành nhiều câu query tối ưu hơn để quét Database.

Hybrid Retrieval: Truy xuất top 15-20 chunks từ Qdrant (kết hợp điểm ngữ nghĩa và điểm từ khóa).

Auto-Merging (Parent-Child): Nếu hệ thống lấy ra quá nhiều "Khoản" thuộc cùng một "Điều", nó sẽ tự động gộp lại và trả về toàn bộ nội dung của "Điều" đó (Parent Node).

Reranking: Đưa top 20 chunks này qua mô hình Cross-Encoder để chấm điểm lại mức độ liên quan thực sự với câu hỏi. Giữ lại top 3 - 5 chunks (Điều luật) xuất sắc nhất.

Bước 5: Thiết kế Prompt & Structured Generation
Đây là bước quyết định việc định dạng đầu ra có thỏa mãn hệ thống chấm điểm của BTC hay không.

Công cụ: Pydantic (để định nghĩa schema), LLM Function Calling.

Thực thi:

Định nghĩa Schema Pydantic cho Output:

Python
from pydantic import BaseModel
class LegalAnswer(BaseModel):
    answer: str
    relevant_docs: list[str]      # Phải ép định dạng: <mã văn bản>|<tên văn bản>
    relevant_articles: list[str]  # Phải ép định dạng: <mã văn bản>|<tên văn bản>|<điều>
System Prompt Master: Cung cấp Role (Chuyên gia luật), Context (Top 5 Điều luật từ bước Retrieval, kèm rõ metadata), và Instruction (Buộc trả lời dựa trên context, cách hành văn rõ ràng, thực tiễn).

Ép LLM (như Qwen2.5 hoặc Gemini) output thẳng ra schema LegalAnswer. Code sẽ tự động trích xuất các metadata đã gắn sẵn trong các chunks được chọn để điền vào relevant_docs và relevant_articles, thay vì bắt LLM tự sinh (để chống ảo giác).

Bước 6: Kiểm thử (Evaluation) & Xuất kết quả
Công cụ: Pandas, tqdm (thanh tiến trình).

Thực thi:

Viết một hàm process_question(id, question) đóng gói toàn bộ Pipeline từ Giai đoạn 4 và 5.

Đọc file input chứa 1591 câu hỏi của BTC.

Dùng vòng lặp (loop) qua từng câu hỏi. Bắt lỗi (Try-Catch) cẩn thận để nếu một câu bị lỗi, hệ thống vẫn chạy tiếp câu sau, gán giá trị mặc định tránh làm hỏng cấu trúc file.

Ghi kết quả vào một danh sách từ điển (List of Dictionaries).

Sử dụng json.dump() để lưu thành file results.json chuẩn.

Chạy lệnh zip tự động: zip submission.zip results.json.

💡 "SECRET SAUCE" - ĐIỂM ĂN TIỀN CHO CUỘC THI NÀY
Rule-based Extraction cho Metadata: Đừng dùng AI để điền mảng relevant_articles. Hãy lập trình (code) lấy trực tiếp metadata của các chunks (đã lấy được sau bước Rerank) và format bằng chuỗi f"{metadata['mã_văn_bản']}|{metadata['tên_văn_bản']}|{metadata['điều']}". Đảm bảo độ chuẩn xác 100% (Tiêu chí 1 của BTC).

Xử lý câu hỏi Yes/No: Trong System Prompt, đối với dạng câu hỏi xác nhận, hãy yêu cầu LLM mở đầu câu trả lời bằng "Có." hoặc "Không.", sau đó mới giải thích căn cứ. Điều này tăng tính thực tiễn và rõ ràng (Tiêu chí 4, 5).
