"""
Baseline RAG Pipeline.
Implements standard Always-Retrieve RAG:
Query -> VectorRetriever -> Context Construction -> Qwen -> Grounded Answer.
"""

import time
from typing import Dict, Any, List, Optional

from src.retrieval.retriever import VectorRetriever
from src.core.llm_loader import LLMManager
from config.config import config, AppConfig


class BaselineRAGPipeline:
    """Standard Baseline RAG pipeline performing unconditional retrieval and generation."""

    def __init__(
        self,
        app_config: AppConfig = config,
        retriever: Optional[VectorRetriever] = None,
        llm_manager: Optional[LLMManager] = None
    ):
        self.config = app_config
        self.retriever = retriever or VectorRetriever(cfg=self.config.retrieval)
        self.llm_manager = llm_manager or LLMManager.get_instance(cfg=self.config.llm)

    def run(self, query: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """
        Executes the Baseline RAG pipeline:
        1. Retrieve top-k documents from FAISS CPU index.
        2. Format context block with document sources.
        3. Pass context and query to Qwen on GPU.
        4. Return response with metadata and latency measurements.
        """
        start_total = time.perf_counter()

        # Step 1: Retrieval
        start_retrieval = time.perf_counter()
        k = top_k if top_k is not None else self.config.retrieval.top_k
        retrieved_chunks = self.retriever.retrieve(query=query, top_k=k)
        retrieval_time_ms = (time.perf_counter() - start_retrieval) * 1000

        # Step 2: Context Construction
        context_str = self.retriever.format_context(retrieved_chunks)

        system_prompt = (
            "You are a helpful and precise assistant. "
            "Answer the user's question using ONLY the provided context. "
            "Do not invent facts or assume information not supported by the context. "
            "If the answer cannot be determined from the context, state that clearly."
        )

        user_prompt = f"Context:\n{context_str}\n\nQuestion: {query}\n\nAnswer:"

        # Step 3: LLM Generation
        start_gen = time.perf_counter()
        answer = self.llm_manager.generate(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            max_new_tokens=self.config.llm.max_new_tokens,
            temperature=self.config.llm.temperature
        )
        generation_time_ms = (time.perf_counter() - start_gen) * 1000

        total_time_ms = (time.perf_counter() - start_total) * 1000

        scores = [chunk.get("score", 0.0) for chunk in retrieved_chunks]

        return {
            "query": query,
            "answer": answer,
            "retrieved_documents": retrieved_chunks,
            "similarity_scores": scores,
            "retrieval_count": len(retrieved_chunks),
            "latency_ms": {
                "retrieval": round(retrieval_time_ms, 2),
                "generation": round(generation_time_ms, 2),
                "total": round(total_time_ms, 2)
            }
        }
