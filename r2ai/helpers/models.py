import os
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
COLLECTION_NAME = "legal_vn_articles"

class RetrievalModels:
    def __init__(self):
        self.qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60.0)
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={"device": "cpu"}, # Đổi sang cuda nếu có
            encode_kwargs={"normalize_embeddings": True}
        )
        self.vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=COLLECTION_NAME,
            embedding=self.embedding_model
        )
        # Reranker model
        print("Loading CrossEncoder Reranker...")
        self.reranker = CrossEncoder('BAAI/bge-reranker-v2-m3', max_length=512, device='cpu')

    def get_vector_store(self):
        return self.vector_store

    def get_reranker(self):
        return self.reranker
