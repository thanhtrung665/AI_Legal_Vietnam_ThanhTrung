from markitdown import MarkItDown
from google import genai 
import os
import time
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOOGLE_API_KEY:
    raise ValueError("Không tìm thấy GOOGLE API KEY")

client = genai.Client(api_key=GOOGLE_API_KEY)
md_local = MarkItDown()

# Tạo hàm 
def process_pdf_with_gemini(pdf_path):
    print(f"Đang upload {os.path.basename(pdf_path)}' lên Google Cloud")
    uploaded_file = client.files.upload(file=pdf_path)
    print(f"Chờ AI phân tích cấu trúc file")
    time.sleep(5)

    print("Gemini trích xuất markdown....")

    prompt = """
    Bạn là chuyên gia số hóa tài liệu pháp lý. Hãy trích xuất TOÀN BỘ nội dung của tài liệu PDF này 
    và trình bày lại dưới định dạng Markdown chuẩn.
    Yêu cầu:
    1. Giữ nguyên cấu trúc phân cấp (Chương, Điều, Khoản, Điểm) bằng thẻ Heading (#, ##, ###).
    2. Nếu có Bảng biểu, hãy tạo bảng Markdown tương ứng.
    3. Trả về nội dung text duy nhất, không giải thích hay bình luận thêm.
    """

    respone = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[uploaded_file,prompt]
    )
    client.files.delete(name=uploaded_file.name)
    return response.text

def convert_to_markdown(input_path, output_path):
    os.makedirs(output_folder, exist_ok=True)
    files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.pdf', 'docx', 'doc'))]
    print(f"Tìm thấy {len(files)} file cần xử lý")

    for filename in files:
        file_path = os.path.join(input_folder, filename)
        base_name = os.path.splitext(filename)[0]
        ext = os.path.splitext(filename)[1].lower()
        output_path = os.path.join(output_folder, f"{base_name}.md")

        if os.path.exists(output_path) and os.path.getsize(output_path) > 100:
            print(f"⏭️ Bỏ qua {filename} (Đã có sẵn file Markdown).")
            continue
            
        print(f"\n🔄 Đang xử lý: {filename}")
        
        if ext == '.pdf':
            # --- CƠ CHẾ TỰ ĐỘNG THỬ LẠI (AUTO-RETRY) ---
            max_retries = 3  # Cho phép thử lại tối đa 3 lần nếu bị lỗi quá tải
            for attempt in range(max_retries):
                try:
                    markdown_text = process_pdf_with_gemini(file_path)
                    
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(markdown_text)
                    print(f"✅ Thành công: Đã lưu {base_name}.md")
                    
                    # Nghỉ 15 giây (thay vì 5) để nhịp độ gọi API đều đặn, không bị Google phạt
                    print("  💤 Tạm nghỉ 15 giây để làm mát API...")
                    time.sleep(15)
                    break # Thoát khỏi vòng lặp Retry vì file đã xử lý thành công
                    
                except Exception as e:
                    error_msg = str(e)
                    # Nếu Google báo quá tải (429 hoặc Quota)
                    if "429" in error_msg or "Quota" in error_msg:
                        wait_time = 60 # Nghỉ hẳn 1 phút để Google khôi phục hạn mức
                        print(f"  ⚠️ Bị tuýt còi vì quá tải API! Hệ thống tự động nghỉ {wait_time} giây để thử lại (Lần {attempt + 1}/{max_retries})...")
                        time.sleep(wait_time)
                    else:
                        print(f"❌ Lỗi không xác định khi xử lý {filename}: {e}")
                        break # Nếu là lỗi khác (file hỏng...) thì bỏ qua file này
                        
        elif ext in ['.docx', '.doc']:
            try:
                print("  ⚡ Dùng công cụ Local để xử lý file Word...")
                result = md_local.convert(file_path)
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(result.text_content)
                print(f"✅ Thành công: Đã lưu {base_name}.md")
            except Exception as e:
                print(f"❌ Lỗi khi xử lý {filename}: {e}")

if __name__ == "__main__":
    INPUT_DIR = "./AI_Legal/r2ai/data/Vietnam_Legal"
    OUTPUT_DIR = "./AI_Legal/r2ai/data/markdown_docs"
    
    batch_convert_docs(INPUT_DIR, OUTPUT_DIR)
    print("\n🎉 HOÀN TẤT CHIẾN DỊCH TRÍCH XUẤT!")