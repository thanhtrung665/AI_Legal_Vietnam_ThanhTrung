import os
import json
import re
from pathlib import Path
from typing import List

# Cố gắng import LlamaIndex, nếu chưa có sẽ báo lỗi để người dùng cài đặt
try:
    from llama_index.core.node_parser import MarkdownNodeParser
    from llama_index.core import Document
    from llama_index.core.schema import BaseNode
except ImportError:
    raise ImportError("Vui lòng cài đặt llama-index-core: pip install llama-index-core")

# Cấu hình đường dẫn
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "markdown_data"
METADATA_MAPPING_PATH = BASE_DIR.parent / "metadata_mapping.json"

def load_metadata_mapping() -> dict:
    """
    Tải file metadata mapping (chứa mã văn bản và tên văn bản).
    """
    with open(METADATA_MAPPING_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

def process_markdown_to_nodes(file_path: str) -> List[BaseNode]:
    """
    Đọc file markdown, sử dụng MarkdownNodeParser để chia nhỏ (chunk) theo heading,
    và gán metadata tuyệt đối (mã_văn_bản, tên_văn_bản, điều) cho mỗi chunk.
    
    Args:
        file_path (str): Đường dẫn tuyệt đối tới file markdown.
        
    Returns:
        List[BaseNode]: Danh sách các node đã chứa metadata.
    """
    file_path_obj = Path(file_path)
    file_stem = file_path_obj.stem
    
    # 1. Tải metadata gốc của văn bản
    metadata_mapping = load_metadata_mapping()
    base_metadata = metadata_mapping.get(file_stem, {
        "mã_văn_bản": "Unknown",
        "tên_văn_bản": "Unknown"
    })
    
    # 2. Đọc nội dung Markdown
    with open(file_path_obj, 'r', encoding='utf-8') as f:
        text = f.read()
        
    doc = Document(text=text)
    
    # 3. Sử dụng MarkdownNodeParser để chia nhỏ văn bản tự động
    parser = MarkdownNodeParser()
    nodes = parser.get_nodes_from_documents([doc])
    
    processed_nodes = []
    
    # Tối ưu Regex: Bắt buộc chữ "Điều" phải nằm ở đầu chuỗi (bỏ qua khoảng trắng hoặc dấu #)
    # Dùng ^ để neo đầu dòng
    dieu_pattern = re.compile(r'(?i)^[\s#]*(Điều\s+\d+[a-zA-Z]*)(?:\.|:|\s|$)')
    
    for node in nodes:
        dieu_name = ""
        
        # 1. Tìm trong Header Metadata
        header_keys = [k for k in node.metadata.keys() if str(k).startswith("Header")]
        for key in sorted(header_keys, reverse=True):
            header_val = str(node.metadata[key]).strip()
            # Trong metadata của LlamaIndex, Header_val thường chỉ là text thuần (ví dụ: "Điều 15. Phạm vi")
            # Ta có thể dùng match() hoặc search() đều được, nhưng tốt nhất vẫn ép nó ở đầu
            match = re.match(r'(?i)^[\s]*(Điều\s+\d+[a-zA-Z]*)(?:\.|:|\s|$)', header_val)
            if match:
                dieu_name = header_val.replace("#", "").strip()
                break
                
        # 2. Nếu không thấy trong Metadata, tìm trực tiếp trong Text của Node
        if not dieu_name:
            for line in node.text.split('\n'):
                line = line.strip()
                if not line:
                    continue # Bỏ qua dòng trống
                
                # Chỉ lấy nếu nó nằm CHÍNH XÁC ở đầu dòng
                if dieu_pattern.match(line):
                    dieu_name = line.replace("#", "").strip()
                    break
                
        # 3. Gán metadata tuyệt đối
        node.metadata["mã_văn_bản"] = base_metadata.get("mã_văn_bản", "Unknown")
        node.metadata["tên_văn_bản"] = base_metadata.get("tên_văn_bản", "Unknown")
        node.metadata["điều"] = dieu_name
        
        # 4. TỐI ƯU VECTOR DB: Xóa rác metadata
        for k in header_keys:
            node.metadata.pop(k, None)
            
        processed_nodes.append(node)
        
    return processed_nodes

def process_all_markdowns(data_dir: str) -> List[BaseNode]:
    """
    Xử lý tất cả các file markdown trong một thư mục và gộp thành một danh sách Node duy nhất.
    """
    all_nodes = []
    data_path = Path(data_dir)
    md_files = list(data_path.glob("*.md"))
    
    print(f"[INFO] Bắt đầu xử lý {len(md_files)} file Markdown từ {data_dir}...")
    for md_file in md_files:
        try:
            nodes = process_markdown_to_nodes(str(md_file))
            all_nodes.extend(nodes)
        except Exception as e:
            print(f"[WARNING] Bỏ qua file {md_file.name} do lỗi: {e}")
            
    return all_nodes

if __name__ == "__main__":
    print("="*50)
    print("CHẠY THẬT DATA PIPELINE: XỬ LÝ TOÀN BỘ MARKDOWN")
    print("="*50)
    
    if DATA_DIR.exists():
        all_nodes = process_all_markdowns(str(DATA_DIR))
        print(f"\n[SUCCESS] Đã xử lý xong. Tổng số nodes (chunks) tạo ra từ toàn bộ dữ liệu: {len(all_nodes)}")
        
        print("\n[INFO] Lấy mẫu 3 nodes ngẫu nhiên để kiểm tra metadata:")
        import random
        if all_nodes:
            sample_nodes = random.sample(all_nodes, min(3, len(all_nodes)))
            for i, n in enumerate(sample_nodes, 1):
                print("-" * 40)
                print(f"Node {i}:")
                print(f"  - Mã văn bản : {n.metadata.get('mã_văn_bản')}")
                print(f"  - Tên văn bản: {n.metadata.get('tên_văn_bản')}")
                print(f"  - Điều       : {n.metadata.get('điều')}")
                print(f"  - Nội dung   : {n.text[:150]}...")
            print("-" * 40)
    else:
        print(f"[ERROR] Không tìm thấy thư mục: {DATA_DIR}")
