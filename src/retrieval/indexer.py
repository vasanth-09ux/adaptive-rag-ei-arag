"""
Document Indexer Module for Vector Retrieval.
Chunks text documents, generates BGE embeddings on CPU, and builds a FAISS CPU index.
"""

import json
from pathlib import Path
from typing import List, Dict, Any
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config.config import config, RAW_CORPUS_DIR, RetrievalConfig


class DocumentIndexer:
    """Handles text chunking, embedding generation on CPU, and FAISS index persistence."""

    def __init__(self, cfg: RetrievalConfig = config.retrieval):
        self.cfg = cfg
        self.model = SentenceTransformer(self.cfg.embedding_model_id, device=self.cfg.device)

    @staticmethod
    def chunk_text(
        text: str,
        doc_name: str,
        chunk_size: int = 256,
        chunk_overlap: int = 32
    ) -> List[Dict[str, Any]]:
        """
        Splits text into overlapping chunks deterministically.
        Chunk size and overlap are measured in words to preserve natural word boundaries.
        """
        words = text.split()
        if not words:
            return []

        chunks = []
        step = max(1, chunk_size - chunk_overlap)
        chunk_idx = 0

        for i in range(0, len(words), step):
            chunk_words = words[i : i + chunk_size]
            chunk_str = " ".join(chunk_words)
            chunks.append({
                "chunk_id": chunk_idx,
                "doc_name": doc_name,
                "text": chunk_str,
                "word_count": len(chunk_words),
                "char_count": len(chunk_str)
            })
            chunk_idx += 1
            if i + chunk_size >= len(words):
                break

        return chunks

    def load_and_chunk_corpus(
        self,
        corpus_dir: Path = RAW_CORPUS_DIR
    ) -> List[Dict[str, Any]]:
        """Reads all .txt and .md files in the corpus directory and generates chunks."""
        corpus_path = Path(corpus_dir)
        if not corpus_path.exists():
            raise FileNotFoundError(f"Corpus directory not found: {corpus_path}")

        all_chunks = []
        doc_files = sorted(list(corpus_path.glob("*.txt")) + list(corpus_path.glob("*.md")))

        for doc_file in doc_files:
            with open(doc_file, "r", encoding="utf-8-sig") as f:
                content = f.read().strip()
            if content:
                doc_chunks = self.chunk_text(
                    text=content,
                    doc_name=doc_file.name,
                    chunk_size=self.cfg.chunk_size,
                    chunk_overlap=self.cfg.chunk_overlap
                )
                all_chunks.extend(doc_chunks)

        return all_chunks

    def build_and_save_index(
        self,
        corpus_dir: Path = RAW_CORPUS_DIR,
        index_path: Path | None = None,
        metadata_path: Path | None = None
    ) -> Dict[str, Any]:
        """
        Extracts chunks, computes normalized CPU embeddings, builds FAISS IndexFlatIP,
        and persists index + metadata.
        """
        target_index_path = Path(index_path or self.cfg.index_path)
        target_metadata_path = Path(metadata_path or self.cfg.metadata_path)

        # Ensure parent directories exist
        target_index_path.parent.mkdir(parents=True, exist_ok=True)
        target_metadata_path.parent.mkdir(parents=True, exist_ok=True)

        chunks = self.load_and_chunk_corpus(corpus_dir)
        if not chunks:
            raise ValueError(f"No text documents found to index in {corpus_dir}")

        texts = [chunk["text"] for chunk in chunks]

        # Generate normalized embeddings on CPU (Cosine Similarity via Inner Product)
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
            device=self.cfg.device
        ).astype(np.float32)

        dim = embeddings.shape[1]

        # Build FAISS Flat Inner Product Index (Equivalent to Cosine Similarity for normalized vectors)
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        # Save FAISS index
        faiss.write_index(index, str(target_index_path))

        # Save metadata
        with open(target_metadata_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)

        return {
            "total_chunks": len(chunks),
            "embedding_dim": dim,
            "index_path": str(target_index_path),
            "metadata_path": str(target_metadata_path)
        }
