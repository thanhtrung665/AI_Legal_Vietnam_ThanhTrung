from typing import List, Dict
from rank_bm25 import BM25Okapi
import numpy as np

# Giả sử chúng ta có tập corpus nạp vào BM25 lúc khởi động (đối với production có thể dùng ElasticSearch/Qdrant Keyword)
# Ở đây implement RRF (Reciprocal Rank Fusion)
def reciprocal_rank_fusion(dense_results: List[Dict], sparse_results: List[Dict], k=60):
    """
    Kết hợp kết quả từ Dense và Sparse retrieval sử dụng RRF
    Input lists contain dicts with 'id', 'score', 'doc'
    """
    fused_scores = {}
    docs_map = {}
    
    # Process dense
    for rank, item in enumerate(dense_results):
        doc_id = item['id']
        docs_map[doc_id] = item['doc']
        fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1 / (k + rank + 1)
        
    # Process sparse
    for rank, item in enumerate(sparse_results):
        doc_id = item['id']
        docs_map[doc_id] = item['doc']
        fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1 / (k + rank + 1)
        
    # Sort
    sorted_fused = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return [docs_map[doc_id] for doc_id, _ in sorted_fused]

class LegalRAGPipeline:
    def __init__(self, models):
        self.vector_store = models.get_vector_store()
        self.reranker = models.get_reranker()
        
    def retrieve(self, query: str, top_k_retrieval=20, top_k_rerank=5):
        """
        Quy trình chuẩn: Dense Retrieval -> (Có thể thêm BM25) -> Reranking
        Vì hệ thống Qdrant có hỗ trợ, ta sẽ ưu tiên Dense trước.
        Nếu muốn Hybrid thực sự cần kết nối với index BM25.
        """
        print(f"Retrieving for query: {query}")
        
        # 1. Dense Retrieval
        dense_docs = self.vector_store.similarity_search_with_score(query, k=top_k_retrieval)
        
        # Tạm thời giả lập Sparse để thể hiện logic Hybrid (Trong thực tế cần index riêng)
        # Bỏ qua RRF để rút gọn demo, đi thẳng vào Reranking với Dense docs
        docs_to_rerank = [doc for doc, score in dense_docs]
        
        # 2. Reranking
        pairs = [[query, doc.page_content] for doc in docs_to_rerank]
        rerank_scores = self.reranker.predict(pairs)
        
        # Kết hợp điểm và sắp xếp
        scored_docs = list(zip(docs_to_rerank, rerank_scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        top_docs = [doc for doc, score in scored_docs[:top_k_rerank]]
        return top_docs
