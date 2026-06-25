import re
from datasets import load_dataset
import pandas as pd

def load_and_prepare_data():
    """
    Tải dữ liệu từ Hugging Face và thực hiện tiền xử lý
    """
    print("Loading dataset from Hugging Face...")
    dataset = load_dataset("tmquan/phapdien-moj-gov-vn", split="train")
    
    # Lọc các chủ đề liên quan (tùy chọn theo đề bài)
    # 12: Doanh nghiệp, HTX, 33: Thuế, 20: Lao động, 9: Dân sự, 34: Thương mại
    target_topics = ["12", "33", "20", "9", "34"]
    
    processed_data = []
    
    print("Processing and cleaning data...")
    for row in dataset:
        topic_id = str(row.get('topic_id', ''))
        
        # Nếu muốn lọc theo chủ đề thì dùng đoạn này
        # if topic_id not in target_topics:
        #     continue
            
        # Extract Metadata
        topic_name = row.get('topic_name', '')
        heading_name = row.get('heading_name', '')
        chapter_name = row.get('chapter_name', '')
        article_title = row.get('article_title', '') # vd: Điều 1.1.LQ.1
        content = row.get('content', '')
        source_note = row.get('source_note', '')
        
        # Parsing source note để lấy mã văn bản gốc (Law ID)
        # Ví dụ source note: "Theo Điều 1 Luật Trọng tài thương mại 2010" -> Law ID: Luật Trọng tài thương mại 2010
        # Hoặc cần regex phức tạp hơn dựa trên dữ liệu thực tế. Ở đây là một regex giả định phổ biến:
        law_id = "Không rõ"
        law_name = "Không rõ"
        article_original = article_title
        
        # Thử tìm Mã văn bản (vd: 59/2020/QH14 hoặc Nghị định 123/2020/NĐ-CP)
        law_pattern = r'([A-Za-zĐđ]+ số \d+/\d+/[A-Za-zĐđ\-]+|Luật [A-Za-zĐđ\s]+ \d{4}|Nghị định \d+/\d+/[A-Za-zĐđ\-]+)'
        match_law = re.search(law_pattern, source_note)
        if match_law:
            law_id = match_law.group(1)
            law_name = law_id # Tạm dùng mã làm tên
            
        # Thử tìm tên điều gốc trong source note
        article_pattern = r'(Điều \d+)'
        match_article = re.search(article_pattern, source_note)
        if match_article:
            article_original = match_article.group(1)
        
        # Context Enrichment (Bổ sung ngữ cảnh)
        enriched_content = f"[Chủ đề: {topic_name}] - [Đề mục: {heading_name}] - [Chương: {chapter_name}]\n{article_original}: {content}"
        
        processed_data.append({
            "law_id": law_id,
            "law_name": law_name,
            "article_id": article_original,
            "content": enriched_content,
            "source_note": source_note
        })
        
    df = pd.DataFrame(processed_data)
    print(f"Processed {len(df)} articles.")
    return df

if __name__ == "__main__":
    df = load_and_prepare_data()
    print(df.head())
