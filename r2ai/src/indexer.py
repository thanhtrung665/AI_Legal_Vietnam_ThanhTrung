import os
import torch
from pathlib import Path
from typing import List

try:
    from dotenv import load_dotenv
    from qdrant_client import QdrantClient
    from llama_index.core import VectorStoreIndex, StorageContext, Settings
    from llama_index.vector_stores.qdrant import QdrantVectorStore
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.core.schema import BaseNode
except ImportError:
    raise ImportError("Vui lòng cài đặt: pip install python-dotenv qdrant-client llama-index-vector-stores-qdrant llama-index-embeddings-huggingface fastembed")

BASE_DIR = Path(__file__).resolve().parent.parent
QDRANT_DATA_DIR = BASE_DIR / "data" / "qdrant_db"

# Load environment variables
load_dotenv(BASE_DIR.parent / ".env")

def init_settings():
    """Khởi tạo cấu hình LlamaIndex tối ưu VRAM."""
    
    # [FIX OOM]: Ép Embedding chạy trên CPU để nhường 12GB VRAM cho LLM Qwen
    device = "cpu"
    print(f"[INFO] Khởi tạo Embedding Model (BGE-M3) trên thiết bị: {device.upper()} (Tiết kiệm VRAM)")
    
    # Set HF Token from .env if available
    hf_token = os.getenv("HUGGINGFACE_TOKEN_API")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-m3",
        trust_remote_code=True,
        device=device,
        model_kwargs={"torch_dtype": torch.float16} if device == "cuda" else {} # Giảm 1/2 VRAM trên GPU
    )
    Settings.chunk_size = 512

def build_qdrant_index(nodes: List[BaseNode], collection_name: str = "legal_vn", overwrite: bool = False) -> VectorStoreIndex:
    """
    Tạo hoặc kết nối Qdrant. 
    Lưu ý: Nếu overwrite=True, sẽ xóa Collection cũ để tránh duplicate data.
    """
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    
    if qdrant_url and qdrant_api_key:
        print(f"[INFO] Kết nối Qdrant Cloud tại {qdrant_url}")
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    else:
        print(f"[INFO] Kết nối Qdrant Local tại {QDRANT_DATA_DIR}")
        os.makedirs(QDRANT_DATA_DIR, exist_ok=True)
        client = QdrantClient(path=str(QDRANT_DATA_DIR))
    
    # Xử lý làm sạch collection nếu có yêu cầu ghi đè
    if overwrite and client.collection_exists(collection_name):
        print(f"[WARNING] Xóa Collection cũ '{collection_name}' để build lại từ đầu...")
        client.delete_collection(collection_name)
    
    # enable_hybrid=True tự động dùng FastEmbed sinh Sparse Vector
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        enable_hybrid=True,
        batch_size=32 # Tối ưu tốc độ insert
    )
    
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    if nodes and (overwrite or not client.collection_exists(collection_name)):
        print(f"[INFO] Bắt đầu Embedding & Insert {len(nodes)} nodes (Dense + Sparse)...")
        index = VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            show_progress=True
        )
    else:
        print(f"[INFO] Load existing index từ Collection '{collection_name}'...")
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        
    return index

if __name__ == "__main__":
    from data_pipeline import process_all_markdowns
    
    print("="*50)
    print("CHẠY THẬT PHASE 2: INDEXING TOÀN BỘ VECTOR DB")
    print("="*50)
    
    try:
        init_settings()
        data_dir = BASE_DIR / "data" / "markdown_data"
        
        if data_dir.exists():
            nodes = process_all_markdowns(str(data_dir))
            
            # Ghi đè database cũ để tạo database hoàn chỉnh mới nhất
            print(f"[INFO] Bắt đầu Indexing toàn bộ {len(nodes)} chunks...")
            index = build_qdrant_index(nodes, collection_name="legal_vn_test", overwrite=True)
            
            print(f"[SUCCESS] Đã tạo VectorDB hoàn chỉnh tại {QDRANT_DATA_DIR}")
            
            # Test truy hồi thử 1 câu sau khi build xong
            retriever = index.as_retriever(
                similarity_top_k=3, 
                vector_store_query_mode="hybrid",
                alpha=0.5 
            )
            
            query = "Doanh nghiệp siêu nhỏ là gì?"
            print(f"\n[INFO] Test Truy xuất Hybrid Search cho câu hỏi: '{query}'")
            results = retriever.retrieve(query)
            
            for i, res in enumerate(results, 1):
                print(f"\n[TOP {i}] Score: {res.score:.4f} | Điều: {res.node.metadata.get('điều')}")
                print(f"Content: {res.node.text[:150]}...")
        else:
            print(f"[ERROR] Không tìm thấy thư mục {data_dir}")
            
    except Exception as ex:
        print(f"[ERROR] Quá trình chạy thất bại: {ex}")