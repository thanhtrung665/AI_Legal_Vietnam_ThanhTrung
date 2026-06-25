## **1\. Tổng quan cuộc thi (Overview)**

* **Chủ đề chính:** Tập trung vào bài toán Truy hồi và Hỏi đáp Văn bản Pháp luật Tiếng Việt (Vietnamese Legal Information Retrieval & Question Answering).  
* **Nền tảng tổ chức:** Cuộc thi diễn ra trên nền tảng Chahub, thuộc hệ sinh thái Chahub/Chagrade Benchmark.  
* **Mục tiêu hệ thống cần đạt được:**  
  * Tra cứu chính xác các điều khoản trong Luật Doanh nghiệp và các văn bản liên quan đến doanh nghiệp nhỏ và vừa (SME), đặt ưu tiên cao vào khả năng truy xuất (Retrieval) và căn cứ thông tin (Grounding).  
  * Thấu hiểu sâu sắc ngôn ngữ pháp lý bằng tiếng Việt.  
  * Trích dẫn rõ ràng nguồn tham chiếu (Điều/Khoản/Văn bản) nhằm đảm bảo khả năng kiểm chứng và hạn chế việc phản hồi không có căn cứ pháp lý.  
  * Đưa ra hướng dẫn tư vấn sơ bộ, đi kèm nhắc nhở rủi ro tuân thủ và hiển thị cảnh báo giới hạn của AI.  
  * Kiểm soát chặt chẽ nội dung sai lệch, tuyệt đối tránh hiện tượng AI tự bịa đặt (hallucination) điều luật hoặc nguồn tham chiếu không tồn tại.

## **2\. Quy định về Mô hình & Dữ liệu ngoài (Model & Data Rules)**

* **Giới hạn mô hình LLM:** Bạn được phép sử dụng các mô hình Pretrained và LLM có dữ liệu huấn luyện hoặc công khai với kích thước **dưới 14B**, được phát hành **trước ngày 01/03/2026**.  
* **Yêu cầu tái lập:** Để phục vụ việc kiểm tra và tái lập kết quả, bạn phải cung cấp đầy đủ thông tin về cách thức thu thập/lấy mô hình đó.  
* **Quy định dữ liệu huấn luyện:** Ban Tổ chức **không cung cấp** dữ liệu huấn luyện. Các đội thi được toàn quyền chủ động thu thập, tiền xử lý và khai thác dữ liệu từ các nguồn chính thống (văn bản pháp luật, thông tư, nghị định về thuế, lao động, hợp đồng... liên quan tới SME) hoặc các tập dữ liệu mở (open dataset) phục vụ bài toán Legal NLP.

## **3\. Tiêu chí đánh giá (Evaluation)**

Cuộc thi sử dụng hệ thống **CodaBench** để quản lý các bài nộp và chấm điểm tự động qua hai cấu phần độc lập:

### **A. Đánh giá phần Truy hồi thông tin (Information Retrieval)**

Hệ thống chấm điểm tự động sẽ tìm kiếm các pattern "Điều X" trong các trường kết quả nộp bài, sau đó chuẩn hóa và đối chiếu với đáp án gốc theo định dạng law\_id|tên văn bản|Điều X. Điểm số được tính toán dựa trên các chỉ số sau:

* **Precision (Độ chính xác):**  
  $$Precision \= \\frac{\\text{Trung bình số điều luật truy hồi đúng của mỗi truy vấn}}{\\text{Số điều luật đã truy hồi của mỗi truy vấn}}$$  
* **Recall (Độ bao phủ):**  
  $$Recall \= \\frac{\\text{Trung bình số điều luật truy hồi đúng của mỗi truy vấn}}{\\text{Số điều luật liên quan của mỗi truy vấn}}$$  
* **Chỉ số $F\_2$ (F2 macro average dùng làm điểm đánh giá cuối cùng):**  
  $$F\_2 \= \\frac{5 \\times Precision \\times Recall}{4 \\times Precision \+ Recall}$$

### **B. Đánh giá phần Hỏi đáp pháp luật (Question Answering)**

Được đánh giá toàn diện dựa trên bộ tiêu chí gồm 5 nhóm:

1. **Căn cứ chính xác pháp luật:** Tỷ lệ câu hỏi có ít nhất một điều luật được trích xuất đúng từ câu trả lời (Đánh giá tự động).  
2. **Tính chính xác nội dung:** Mức độ chuẩn xác của câu trả lời so với các quy định pháp luật thực tế.  
3. **Tính đầy đủ & toàn diện:** Khả năng bao quát toàn bộ các khía cạnh liên quan mà câu hỏi đặt ra.  
4. **Tính thực tiễn – khả năng áp dụng:** Khả năng ứng dụng thực tế của câu trả lời trong bối cảnh pháp lý.  
5. **Tính rõ ràng – dễ hiểu:** Cách diễn đạt mạch lạc, dễ tiếp cận cho cả người đọc không chuyên.

## **4\. Cấu trúc bài nộp (Submission Instructions)**

Để hệ thống chấm điểm hoạt động chính xác, bạn cần tuân thủ nghiêm ngặt quy định đóng gói sau:

* **Tên tệp nén:** Phải là **submission.zip**.  
* **Cấu trúc bên trong:** Chỉ chứa duy nhất một tệp tin tên là **results.json** nằm ngay tại thư mục gốc (không được nằm trong bất kỳ thư mục con nào).  
* **Lưu ý quan trọng:** Nếu đặt sai tên tệp results.json, bài nộp sẽ không được chấm điểm. Các bài nộp thiếu câu hỏi hoặc sai định dạng cấu trúc dữ liệu sẽ bị từ chối đánh giá (nhưng không bị trừ vào số lần nộp bài tối đa của bạn).

## **5\. Cấu trúc dữ liệu (Data Schema)**

### **Dữ liệu đầu vào từ Ban Tổ chức**

Hệ thống cung cấp file chứa danh sách câu hỏi bao gồm hai dạng: Câu hỏi xác nhận (Có/Không) và Câu hỏi khai thác thông tin. Mỗi câu hỏi có cấu trúc gồm:

* id: Mã định danh của câu hỏi (kiểu số nguyên integer).  
* question: Nội dung câu hỏi pháp lý (kiểu chuỗi string).

### **Dữ liệu đầu ra yêu cầu trong file results.json**

Bài nộp của bạn phải trả về đầy đủ định dạng cấu trúc JSON chứa các trường sau:

* id: Mã định danh câu hỏi nhận từ BTC.  
* question: Nội dung câu hỏi pháp lý.  
* answer: Nội dung câu trả lời hoàn chỉnh do hệ thống RAG sinh ra.  
* relevant\_docs: Mảng danh sách các văn bản pháp luật liên quan, định dạng: \<mã văn bản\>|\<tên văn bản\>.  
* relevant\_articles: Mảng danh sách các điều luật cụ thể liên quan, định dạng: \<mã văn bản\>|\<tên văn bản\>|\<điều\>.

