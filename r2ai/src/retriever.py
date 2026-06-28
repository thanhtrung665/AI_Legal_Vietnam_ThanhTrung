import torch
from typing import List
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore, TextNode
from sentence_transformers import CrossEncoder

class LegalRetriever:
    """
    Pipeline Truy hồi Nâng cao (Advanced Retrieval Pipeline) - Tối ưu cho Cuộc thi
    """
    
    def __init__(self, index: VectorStoreIndex, retrieve_top_k: int = 20, rerank_top_k: int = 5):
        self.retrieve_top_k = retrieve_top_k
        self.rerank_top_k = rerank_top_k
        
        self.retriever = index.as_retriever(
            similarity_top_k=self.retrieve_top_k, 
            vector_store_query_mode="hybrid",
            alpha=0.5
        )
        
        # [FIX OOM]: Ép Reranker chạy trên CPU để nhường 12GB VRAM cho LLM Qwen
        device = "cpu"
        print(f"[INFO] Khởi tạo BGE-Reranker trên thiết bị: {device.upper()} (Tiết kiệm VRAM)")
        
        model_kwargs = {}
        self.reranker = CrossEncoder(
            "BAAI/bge-reranker-v2-m3", 
            max_length=512, 
            device=device,
            model_kwargs=model_kwargs
        )
        # Đưa Reranker về chế độ eval chống tốn VRAM cho gradients
        self.reranker.model.eval()
        
    def retrieve(self, query: str) -> List[NodeWithScore]:
        """
        Thực hiện truy hồi và trả về danh sách các Điều luật (đã merge).
        """
        # BƯỚC 1: Lấy Top 20 chunks qua Hybrid Search
        hybrid_nodes = self.retriever.retrieve(query)
        if not hybrid_nodes:
            return []
            
        # BƯỚC 2: Reranking với BGE-Reranker-M3
        pairs = [[query, n.node.text] for n in hybrid_nodes]
        
        # [FIX 2]: Dùng Sigmoid để ép điểm Logits về dải xác suất [0, 1]
        import torch.nn as nn
        rerank_scores = self.reranker.predict(pairs, activation_function=nn.Sigmoid())
        
        for node, score in zip(hybrid_nodes, rerank_scores):
            node.score = float(score)
            
        # Sắp xếp lại theo điểm Rerank giảm dần cho 20 chunks
        hybrid_nodes.sort(key=lambda x: x.score, reverse=True)
        
        # [FIX 1]: Đưa TOÀN BỘ 20 chunks vào Auto-Merge trước, KHÔNG cắt Top 5 ở đây!
        merged_nodes = self._auto_merge(hybrid_nodes)
        
        # [OPTIONAL FIX]: Lọc bỏ những "Điều" có điểm quá thấp (Threshold) để chống ảo giác
        # Tùy dữ liệu, bạn có thể thử nghiệm threshold = 0.05 hoặc 0.1
        valid_nodes = [n for n in merged_nodes if n.score >= 0.01] 
        
        # BƯỚC 3: Mới cắt Top K (ví dụ Top 5) từ danh sách ĐÃ GỘP
        return valid_nodes[:self.rerank_top_k]
        
    def _auto_merge(self, nodes: List[NodeWithScore]) -> List[NodeWithScore]:
        """
        Gộp các Khoản (chunk) thành Điều (Parent Node).
        """
        grouped_data = {}
        
        for n in nodes:
            ma_vb = n.node.metadata.get("mã_văn_bản", "Unknown")
            dieu = n.node.metadata.get("điều", "Unknown")
            
            # Khóa gộp (group key)
            key = f"{ma_vb}||{dieu}"
            
            if key not in grouped_data:
                grouped_data[key] = {
                    "metadata": n.node.metadata.copy(),
                    "texts": [n.node.text], # [FIX 3]: Dùng List để lưu text thay vì cộng chuỗi trực tiếp
                    "score": n.score
                }
            else:
                grouped_data[key]["texts"].append(n.node.text)
                # Đại diện điểm của "Điều" là điểm cao nhất của "Khoản" bên trong nó
                grouped_data[key]["score"] = max(grouped_data[key]["score"], n.score)
                
        # Tái tạo lại danh sách NodeWithScore
        merged_nodes = []
        for key, data in grouped_data.items():
            # Join các văn bản lại với nhau bằng dấu "..."
            merged_text = "\n[...]\n".join(data["texts"])
            new_node = TextNode(text=merged_text, metadata=data["metadata"])
            merged_nodes.append(NodeWithScore(node=new_node, score=data["score"]))
            
        # Sắp xếp lại theo thứ tự điểm cao nhất sau khi merge
        merged_nodes.sort(key=lambda x: x.score, reverse=True)
        return merged_nodes


