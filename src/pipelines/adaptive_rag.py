"""
End-to-End Adaptive RAG Pipeline.
Supports:
1. DIRECT mode (No retrieval, direct LLM generation)
2. BASELINE_RAG mode (Always-retrieve RAG)
3. ADAPTIVE mode (EI-ARAG: Layer-1 contextual representation -> Binary Classifier -> Dynamic Routing)
"""

import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import torch

from src.core.llm_loader import LLMManager
from src.core.layer_extractor import Layer1Extractor
from src.retrieval.retriever import VectorRetriever
from src.classifier.model import RetrievalClassifier, SmallRetrievalClassifier
from config.config import config, AppConfig, MODELS_DIR


class AdaptiveRAGPipeline:
    """
    Unified Pipeline supporting DIRECT, BASELINE_RAG, and ADAPTIVE modes.
    """

    def __init__(
        self,
        mode: str = "adaptive",
        classifier_path: Optional[Path] = None,
        threshold: float = 0.5,
        app_config: AppConfig = config,
        llm_manager: Optional[LLMManager] = None,
        retriever: Optional[VectorRetriever] = None,
        extractor: Optional[Layer1Extractor] = None
    ):
        self.mode = mode.lower()
        if self.mode not in ["direct", "baseline_rag", "adaptive"]:
            raise ValueError(f"Invalid mode: {mode}. Choose from 'direct', 'baseline_rag', 'adaptive'.")

        self.config = app_config
        self.threshold = threshold
        self.llm_manager = llm_manager or LLMManager.get_instance(cfg=self.config.llm)
        self.retriever = retriever or VectorRetriever(cfg=self.config.retrieval)
        self.extractor = extractor or Layer1Extractor(app_config=self.config, llm_manager=self.llm_manager)

        # Dynamic classifier loader
        self.classifier_path = Path(classifier_path or self.config.classifier.model_save_path or (MODELS_DIR / "classifier_expanded.pt"))
        self.classifier = self._load_classifier(self.classifier_path)

    def _load_classifier(self, checkpoint_path: Path):
        """Loads trained classifier checkpoint dynamically based on layer dimensions."""
        if not checkpoint_path.exists():
            print(f"Warning: Classifier checkpoint not found at {checkpoint_path}. Using uninitialized SmallRetrievalClassifier.")
            return SmallRetrievalClassifier(input_dim=self.extractor.hidden_size, hidden_dim=32, num_classes=2)

        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

        if "fc3.weight" in state_dict:
            hidden_dim1 = state_dict["fc1.weight"].shape[0]
            hidden_dim2 = state_dict["fc2.weight"].shape[0]
            input_dim = state_dict["fc1.weight"].shape[1]
            model = RetrievalClassifier(input_dim=input_dim, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, num_classes=2)
        else:
            hidden_dim = state_dict["fc1.weight"].shape[0]
            input_dim = state_dict["fc1.weight"].shape[1]
            model = SmallRetrievalClassifier(input_dim=input_dim, hidden_dim=hidden_dim, num_classes=2)

        model.load_state_dict(state_dict)
        model.eval()
        return model

    def _direct_generate(self, query: str) -> Tuple[str, float]:
        """Generates answer using parametric knowledge only."""
        start_gen = time.perf_counter()
        direct_system_prompt = (
            "You are a helpful and concise assistant. "
            "Answer the question directly and accurately in as few words as possible."
        )
        direct_user_prompt = f"Question: {query}\n\nAnswer:"
        answer = self.llm_manager.generate(
            user_prompt=direct_user_prompt,
            system_prompt=direct_system_prompt,
            max_new_tokens=self.config.llm.max_new_tokens,
            temperature=self.config.llm.temperature
        )
        gen_latency_ms = (time.perf_counter() - start_gen) * 1000
        return answer, gen_latency_ms

    def _rag_generate(self, query: str, top_k: Optional[int] = None) -> Tuple[str, List[Dict[str, Any]], float, float]:
        """Performs retrieval and grounded generation."""
        k = top_k if top_k is not None else self.config.retrieval.top_k
        start_ret = time.perf_counter()
        retrieved_chunks = self.retriever.retrieve(query=query, top_k=k)
        ret_latency_ms = (time.perf_counter() - start_ret) * 1000

        context_str = self.retriever.format_context(retrieved_chunks)

        start_gen = time.perf_counter()
        rag_system_prompt = (
            "You are a helpful and precise assistant. "
            "Answer the user's question using ONLY the provided context. "
            "Do not invent facts or assume information not supported by the context. "
            "If the answer cannot be determined from the context, state that clearly."
        )
        rag_user_prompt = f"Context:\n{context_str}\n\nQuestion: {query}\n\nAnswer:"
        answer = self.llm_manager.generate(
            user_prompt=rag_user_prompt,
            system_prompt=rag_system_prompt,
            max_new_tokens=self.config.llm.max_new_tokens,
            temperature=self.config.llm.temperature
        )
        gen_latency_ms = (time.perf_counter() - start_gen) * 1000
        return answer, retrieved_chunks, ret_latency_ms, gen_latency_ms

    def answer(self, query: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """
        Answers the query using the configured mode.
        """
        start_total = time.perf_counter()

        if self.mode == "direct":
            answer_text, gen_lat = self._direct_generate(query)
            total_lat = (time.perf_counter() - start_total) * 1000
            return {
                "query": query,
                "mode": "direct",
                "routing_decision": "DIRECT",
                "classifier_label": None,
                "classifier_probability": None,
                "answer": answer_text,
                "retrieved_documents": [],
                "retrieval_latency_ms": 0.0,
                "generation_latency_ms": round(gen_lat, 2),
                "total_latency_ms": round(total_lat, 2)
            }

        elif self.mode == "baseline_rag":
            answer_text, docs, ret_lat, gen_lat = self._rag_generate(query, top_k=top_k)
            total_lat = (time.perf_counter() - start_total) * 1000
            return {
                "query": query,
                "mode": "baseline_rag",
                "routing_decision": "RAG",
                "classifier_label": None,
                "classifier_probability": None,
                "answer": answer_text,
                "retrieved_documents": docs,
                "retrieval_latency_ms": round(ret_lat, 2),
                "generation_latency_ms": round(gen_lat, 2),
                "total_latency_ms": round(total_lat, 2)
            }

        elif self.mode == "adaptive":
            # 1. Extract Layer-1 representation of raw query
            query_emb = self.extractor.extract_single(query)  # (1536,)
            query_emb_batch = query_emb.unsqueeze(0)  # (1, 1536)

            # 2. Predict routing decision via CPU Classifier
            probs = self.classifier.predict_proba(query_emb_batch)[0]
            prob_label1 = round(probs[1].item(), 4)
            label = int(self.classifier.predict(query_emb_batch, threshold=self.threshold)[0].item())

            routing_decision = "RAG" if label == 1 else "DIRECT"

            # 3. Execute routed generation
            if routing_decision == "RAG":
                answer_text, docs, ret_lat, gen_lat = self._rag_generate(query, top_k=top_k)
                total_lat = (time.perf_counter() - start_total) * 1000
                return {
                    "query": query,
                    "mode": "adaptive",
                    "routing_decision": "RAG",
                    "classifier_label": label,
                    "classifier_probability": prob_label1,
                    "answer": answer_text,
                    "retrieved_documents": docs,
                    "retrieval_latency_ms": round(ret_lat, 2),
                    "generation_latency_ms": round(gen_lat, 2),
                    "total_latency_ms": round(total_lat, 2)
                }
            else:
                answer_text, gen_lat = self._direct_generate(query)
                total_lat = (time.perf_counter() - start_total) * 1000
                return {
                    "query": query,
                    "mode": "adaptive",
                    "routing_decision": "DIRECT",
                    "classifier_label": label,
                    "classifier_probability": prob_label1,
                    "answer": answer_text,
                    "retrieved_documents": [],
                    "retrieval_latency_ms": 0.0,
                    "generation_latency_ms": round(gen_lat, 2),
                    "total_latency_ms": round(total_lat, 2)
                }
