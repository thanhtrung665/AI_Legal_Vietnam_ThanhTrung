# Data Science End-to-End Handbook

Cẩm nang Khoa học Dữ liệu chuẩn Production — từ thu thập dữ liệu, EDA, phân tích thống kê, tiền xử lý, chuẩn hóa, chia dữ liệu, huấn luyện, đánh giá (metrics, confusion matrix, threshold, calibration), giải thích mô hình, trực quan hóa đến MLOps — kèm code mẫu Python cho từng phần.

## Sản phẩm

| File | Mô tả |
|---|---|
| [`Data_Science_End_to_End_Handbook.md`](Data_Science_End_to_End_Handbook.md) | Bản Markdown đầy đủ (ghép từ các chương) |
| [`Data_Science_End_to_End_Handbook.pdf`](Data_Science_End_to_End_Handbook.pdf) | Bản PDF A4 có trang bìa, mục lục, tô màu code |
| [`chapters/`](chapters/) | Từng chương riêng lẻ — chỉnh sửa tại đây |
| [`build/`](build/) | Script ghép Markdown và xuất PDF |

## Mục lục

| # | Chương |
|---|---|
| 0 | [Kế hoạch & tổng quan: vòng đời dự án, cấu trúc repo, môi trường](chapters/00_ke_hoach_va_tong_quan.md) |
| 1 | [Định nghĩa bài toán & thiết kế giải pháp](chapters/01_dinh_nghia_bai_toan.md) |
| 2 | [Thu thập dữ liệu: file, SQL, API, scraping, streaming, data contract, DVC](chapters/02_thu_thap_du_lieu.md) |
| 3 | [Phân tích khám phá dữ liệu (EDA)](chapters/03_eda.md) |
| 4 | [Phân tích dữ liệu & thống kê suy luận, A/B testing](chapters/04_phan_tich_du_lieu.md) |
| 5 | [Tiền xử lý & Feature Engineering](chapters/05_tien_xu_ly.md) |
| 6 | [Chuẩn hóa & biến đổi dữ liệu](chapters/06_chuan_hoa.md) |
| 7 | [Chia dữ liệu & chiến lược validation](chapters/07_chia_du_lieu.md) |
| 8 | [Huấn luyện mô hình, tuning, ensemble, deep learning](chapters/08_huan_luyen_mo_hinh.md) |
| 9 | [Đánh giá mô hình: metrics, confusion matrix, threshold, calibration](chapters/09_danh_gia_mo_hinh.md) |
| 10 | [Giải thích mô hình (XAI) & fairness](chapters/10_giai_thich_mo_hinh.md) |
| 11 | [Trực quan hóa dữ liệu & kết quả](chapters/11_truc_quan_hoa.md) |
| 12 | [MLOps: tracking, registry, serving, Docker, CI/CD, monitoring, retraining](chapters/12_mlops.md) |
| 13 | [Checklist production & anti-patterns](chapters/13_checklist_production.md) |
| — | [Phụ lục: cheatsheet thư viện, chọn metric, chọn thuật toán, thuật ngữ, tham khảo](chapters/14_phu_luc.md) |

## Build lại Markdown + PDF

Yêu cầu: `pandoc` ≥ 3, Node.js và `playwright` (Chromium).

```bash
./docs/data-science-handbook/build/build.sh
```

Script sẽ ghép `chapters/*.md` → `Data_Science_End_to_End_Handbook.md` → HTML (pandoc) → PDF (Chromium headless).
