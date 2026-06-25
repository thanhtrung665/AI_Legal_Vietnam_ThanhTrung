# Nguồn dữ liệu 

1. Nguồn  
   Lấy từ huggingface : https://huggingface.co/datasets/tmquan/phapdien-moj-gov-vn t 

   ### **Bản chất của "Bộ Pháp điển" là gì?**

Pháp điển không phải là một văn bản luật mới. Bản chất của nó là **tập hợp tất cả các điều luật đang còn hiệu lực** thuộc cùng một lĩnh vực, được sắp xếp lại theo một cấu trúc logic (Chủ đề \-\> Đề mục \-\> Chương \-\> Mục \-\> Điều).

* Do đó, dữ liệu ở đây cực kỳ sạch, đã được chuẩn hóa và **100% là văn bản còn hiệu lực** từ Bộ Tư pháp. Bạn sẽ không sợ LLM nạp phải luật cũ đã hết hiệu lực.

  ### **2\. Các thông số kỹ thuật cốt lõi**

* **Quy mô:** Hơn **64,464 Điều luật**, bao phủ 42 Chủ đề và 202 Đề mục.  
* **Cấu trúc phân cấp:** Mỗi dòng dữ liệu đại diện cho **1 Điều (Article-level corpus)**. Đúng chuẩn phương pháp *Article-level Chunking* mà chúng ta đã vạch ra ở Bước 4\!  
* **Các trường dữ liệu (Columns) quan trọng:**  
  * `article_title`: Tiêu đề dạng định danh hệ thống pháp điển (ví dụ: `Điều 1.1.LQ.1`).  
  * `article_anchor`: Mã băm định danh duy nhất.  
  * `content`: Toàn văn nội dung điều luật đã được chuẩn hóa.  
  * **Chú thích nguồn (Source-note):** Đạt tỉ lệ **100%**. Đây là phần ghi chú điều này được trích xuất từ văn bản gốc nào (Luật số mấy, Nghị định nào, ngày ban hành...).  
  * **Liên kết gốc (Source-link):** 100% có link trỏ về trang `vbpl.vn`.

  ## **🔥 Điểm cộng cực lớn (The Pros) cho hệ thống RAG**

* **Chunking sẵn ở cấp độ Điều:** Bạn không cần tốn quá nhiều công sức viết thuật toán regex phức tạp để tự chia nhỏ văn bản từ file PDF lớn nữa. Dataset này đã chia trọn vẹn từng Điều vào từng dòng một cách hoàn hảo.  
* **Ngữ cảnh giàu có (Context-rich):** Nhờ có cấu trúc phân cấp (Chủ đề, Đề mục, Chương), bạn dễ dàng thực hiện **Context Enrichment** bằng cách ép thêm các trường này vào đầu chunk để làm tăng độ chính xác khi Embedding.  
* **Bao phủ toàn bộ miền tri thức:** Với 64k Điều, nó chứa trọn vẹn các mảng về Doanh nghiệp, Thuế, Lao động, Hợp đồng mà cuộc thi yêu cầu.

  ## **⚠️ Điểm cốt tử cần lưu ý (The Catch)**

Dù đây là mỏ vàng, nhưng nếu bê nguyên xi vào hệ thống, bạn sẽ bị **0 điểm** ở phần trích xuất nguồn. Tại sao?

1. **Sai lệch định dạng "Điều":** Trong Bộ Pháp điển, tiêu đề của điều luật được mã hóa lại theo định dạng của Pháp điển (ví dụ: `Điều 1.1.LQ.1`). Trong khi đó, Ban tổ chức cuộc thi yêu cầu kết quả trả về phải là tên Điều gốc của văn bản pháp luật (ví dụ: `Điều 4`, `Điều 5`).  
2. **Nhiệm vụ bóc tách Source-note:** Bạn bắt buộc phải viết một script để parse (phân tích) trường **Source-note** hoặc **Source-link** của dataset này nhằm lấy ra được mã văn bản gốc (như `04/2017/QH14` hay `80/2021/NĐ-CP`) và số hiệu Điều gốc của nó.

3\. Các chủ đề lấy từ dataset

`#12` Doanh nghiệp, HTX

`#33` Thuế, phí, lệ phí...

`#20` Lao động

`#9` Dân sự

`#2` Bảo hiểm

`#17` Kế toán, kiểm toán

`#34` Thương mại  
`#4` Các tổ chức tín dụng   
`#29` Sở hữu trí tuệ  
**`#39` Trọng tài thương mại**  
