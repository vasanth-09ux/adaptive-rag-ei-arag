"""
Retriever Module for Vector Search.
Embeds user queries on CPU using BGE-small and searches the local FAISS CPU index.
"""

import json
from pathlib import Path
from typing import List, Dict, Any
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config.config import config, RetrievalConfig


class VectorRetriever:
    """Retrieves top-k relevant document chunks using FAISS and CPU embeddings."""

    def __init__(
        self,
        cfg: RetrievalConfig = config.retrieval,
        index_path: Path | None = None,
        metadata_path: Path | None = None
    ):
        self.cfg = cfg
        self.index_path = Path(index_path or self.cfg.index_path)
        self.metadata_path = Path(metadata_path or self.cfg.metadata_path)

        if not self.index_path.exists():
            raise FileNotFoundError(f"FAISS index file not found at: {self.index_path}")
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found at: {self.metadata_path}")

        # Load FAISS index
        self.index = faiss.read_index(str(self.index_path))

        # Load metadata
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        # Load CPU embedding model
        self.model = SentenceTransformer(self.cfg.embedding_model_id, device=self.cfg.device)

    def retrieve(self, query: str, top_k: int | None = None) -> List[Dict[str, Any]]:
        """
        Embeds the query on CPU, performs FAISS similarity search, and returns ranked results.
        """
        k = top_k if top_k is not None else self.cfg.top_k
        k = min(k, self.index.ntotal)
        if k <= 0:
            return []

        # Encode and normalize query embedding
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            device=self.cfg.device
        ).astype(np.float32)

        # FAISS search: Inner Product on normalized vectors = Cosine Similarity
        scores, indices = self.index.search(query_embedding, k)

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx == -1:
                continue
            chunk_data = self.metadata[idx].copy()
            chunk_data["score"] = float(score)
            chunk_data["rank"] = rank
            results.append(chunk_data)

        return results

    def format_context(self, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """Formats retrieved chunks into a clean context string for LLM prompting."""
        if not retrieved_chunks:
            return ""
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, start=1):
            source = chunk.get("doc_name", "Unknown")
            text = chunk.get("text", "")
            context_parts.append(f"[Document {i} | Source: {source}]\n{text}")
        return "\n\n".join(context_parts)
