import os
import uuid
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from parser import load_and_prepare_data

# Load environment variables
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
COLLECTION_NAME = "legal_vn_articles"

def get_embeddings():
    """Khởi tạo BGE-M3 embedding model"""
    hf_token = os.getenv("HUGGINGFACE_TOKEN_API")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    model_name = "BAAI/bge-m3"
    model_kwargs = {"device": "cpu"} # Thay bằng "cuda" nếu có GPU
    encode_kwargs = {"normalize_embeddings": True}
    return HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

def setup_qdrant_collection(client):
    """Tạo collection với cấu hình HNSW tối ưu cho BGE-M3 (dim=1024)"""
    try:
        client.get_collection(collection_name=COLLECTION_NAME)
        print(f"Collection '{COLLECTION_NAME}' already exists.")
    except Exception:
        print(f"Creating collection '{COLLECTION_NAME}'...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=qmodels.VectorParams(
                size=1024, # BGE-M3 size
                distance=qmodels.Distance.COSINE
            ),
            # Tối ưu hóa HNSW Index (theo kiến thức từ PDF mentor)
            hnsw_config=qmodels.HnswConfigDiff(
                m=16,
                ef_construct=100
            )
        )

def upload_to_qdrant(df):
    """Embed và upload dữ liệu lên Qdrant"""
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=120.0)
    setup_qdrant_collection(client)
    
    embeddings = get_embeddings()
    qdrant = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings
    )
    
    print(f"Embedding and uploading {len(df)} documents to Qdrant...")
    
    # Langchain yêu cầu chuẩn bị texts và metadatas
    texts = df['content'].tolist()
    metadatas = df[['law_id', 'law_name', 'article_id']].to_dict(orient='records')
    
    # Upload theo batch nhỏ để tránh timeout/OOM
    batch_size = 10
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_metas = metadatas[i:i+batch_size]
        
        qdrant.add_texts(
            texts=batch_texts,
            metadatas=batch_metas
        )
        print(f"Uploaded batch {i//batch_size + 1}/{len(texts)//batch_size + 1}")

    print("Upload complete!")

if __name__ == "__main__":
    df = load_and_prepare_data()
    # Để test nhanh, lấy 100 dòng đầu tiên
    upload_to_qdrant(df.head(100))
