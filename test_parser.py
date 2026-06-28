from langchain_text_splitters import MarkdownHeaderTextSplitter

markdown_text = """
## LUẬT
HỖ TRỢ DOANH NGHIỆP NHỎ VÀ VỪA

### Chương I
NHỮNG QUY ĐỊNH CHUNG

#### Điều 2. Đối tượng áp dụng
1. Doanh nghiệp được thành lập, tổ chức và hoạt động theo quy định của pháp luật về doanh nghiệp, đáp ứng các tiêu chí xác định doanh nghiệp nhỏ và vừa theo quy định của Luật này.
2. Cơ quan, tổ chức và cá nhân liên quan đến hỗ trợ doanh nghiệp nhỏ và vừa.
"""

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
    ("####", "Header 4"),
]

markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
splits = markdown_splitter.split_text(markdown_text)
for split in splits:
    print(split.metadata)
    print(split.page_content)
    print("---")
