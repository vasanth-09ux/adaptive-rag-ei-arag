"""
EI-ARAG Data and Label Generator Module.
Generates binary training labels (y=1 if RAG improves over Direct, else 0)
using deterministic dual-inference (Direct vs. RAG) and substring matching.
"""

import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.core.llm_loader import LLMManager
from src.retrieval.retriever import VectorRetriever
from config.config import config, DATA_DIR, AppConfig


class DataGenerator:
    """Automated label generator following the EI-ARAG (COLING 2025) methodology."""

    def __init__(
        self,
        app_config: AppConfig = config,
        llm_manager: Optional[LLMManager] = None,
        retriever: Optional[VectorRetriever] = None
    ):
        self.config = app_config
        self.llm_manager = llm_manager or LLMManager.get_instance(cfg=self.config.llm)
        self.retriever = retriever or VectorRetriever(cfg=self.config.retrieval)

    @staticmethod
    def is_correct(predicted_answer: str, gold_answer: str) -> bool:
        """
        Determines correctness using normalized case-insensitive substring containment.
        Follows the evaluation criterion in the EI-ARAG paper.
        """
        norm_pred = predicted_answer.strip().lower()
        norm_gold = gold_answer.strip().lower()
        return norm_gold in norm_pred

    def process_example(
        self,
        question: str,
        gold_answer: str,
        top_k: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Performs dual inference for a single QA pair:
        1. Direct generation (Parametric LLM knowledge only)
        2. RAG generation (Retrieved context + LLM)
        3. Evaluates both against gold_answer and computes binary label.
        """
        k = top_k if top_k is not None else self.config.retrieval.top_k

        # 1. Direct Inference (No Retrieval)
        start_direct = time.perf_counter()
        direct_system_prompt = (
            "You are a helpful and concise assistant. "
            "Answer the question directly and accurately in as few words as possible."
        )
        direct_user_prompt = f"Question: {question}\n\nAnswer:"
        direct_answer = self.llm_manager.generate(
            user_prompt=direct_user_prompt,
            system_prompt=direct_system_prompt,
            max_new_tokens=self.config.llm.max_new_tokens,
            temperature=self.config.llm.temperature
        )
        direct_latency_ms = (time.perf_counter() - start_direct) * 1000

        # 2. RAG Inference (With Retrieval)
        start_retrieval = time.perf_counter()
        retrieved_chunks = self.retriever.retrieve(query=question, top_k=k)
        retrieval_latency_ms = (time.perf_counter() - start_retrieval) * 1000
        
        context_str = self.retriever.format_context(retrieved_chunks)

        start_rag_gen = time.perf_counter()
        rag_system_prompt = (
            "You are a helpful and precise assistant. "
            "Answer the user's question using ONLY the provided context. "
            "Do not invent facts or assume information not supported by the context. "
            "If the answer cannot be determined from the context, state that clearly."
        )
        rag_user_prompt = f"Context:\n{context_str}\n\nQuestion: {question}\n\nAnswer:"
        rag_answer = self.llm_manager.generate(
            user_prompt=rag_user_prompt,
            system_prompt=rag_system_prompt,
            max_new_tokens=self.config.llm.max_new_tokens,
            temperature=self.config.llm.temperature
        )
        rag_gen_latency_ms = (time.perf_counter() - start_rag_gen) * 1000
        total_rag_latency_ms = retrieval_latency_ms + rag_gen_latency_ms

        # 3. Correctness Assessment
        direct_correct = self.is_correct(direct_answer, gold_answer)
        rag_correct = self.is_correct(rag_answer, gold_answer)

        # 4. EI-ARAG Label Assignment Criterion
        # Label = 1 ONLY when RAG succeeds AND Direct fails
        label = 1 if (rag_correct and not direct_correct) else 0

        sources = [
            {
                "doc_name": chunk.get("doc_name"),
                "chunk_id": chunk.get("chunk_id"),
                "score": round(chunk.get("score", 0.0), 4)
            }
            for chunk in retrieved_chunks
        ]

        record: Dict[str, Any] = {}
        if metadata:
            record.update(metadata)

        record.update({
            "question": question,
            "gold_answer": gold_answer,
            "direct_answer": direct_answer,
            "rag_answer": rag_answer,
            "direct_correct": direct_correct,
            "rag_correct": rag_correct,
            "label": label,
            "retrieved_sources": sources,
            "retrieval_count": len(retrieved_chunks),
            "latency_ms": {
                "direct_generation": round(direct_latency_ms, 2),
                "retrieval": round(retrieval_latency_ms, 2),
                "rag_generation": round(rag_gen_latency_ms, 2),
                "rag_total": round(total_rag_latency_ms, 2)
            }
        })

        return record

    def generate_labeled_dataset(
        self,
        qa_pairs: List[Dict[str, Any]],
        output_file: Optional[Path] = None,
        top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Processes a list of QA pairs, incrementally writes labeled records to JSON,
        supports automatic resume of already-completed items, and computes metrics.
        """
        target_path = Path(output_file or (DATA_DIR / "labeled_dataset.json"))
        target_path.parent.mkdir(parents=True, exist_ok=True)

        labeled_records: List[Dict[str, Any]] = []
        completed_ids = set()

        if target_path.exists():
            try:
                with open(target_path, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
                if isinstance(existing_data, list):
                    labeled_records = existing_data
                    for r in labeled_records:
                        r_id = r.get("id") or r.get("question")
                        if r_id:
                            completed_ids.add(r_id)
                    print(f"Resuming: loaded {len(labeled_records)} existing records from {target_path.name}.")
            except Exception as e:
                print(f"Warning: could not parse existing file {target_path} ({e}). Starting fresh.")
                labeled_records = []
                completed_ids = set()

        def _save_checkpoint(records: List[Dict[str, Any]]):
            tmp_file = target_path.with_suffix(".tmp")
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2, ensure_ascii=False)
            tmp_file.replace(target_path)

        for i, item in enumerate(qa_pairs, start=1):
            item_id = item.get("id") or item.get("question")
            if item_id in completed_ids:
                print(f"[{i}/{len(qa_pairs)}] Skipping already completed: {item_id}")
                continue

            print(f"[{i}/{len(qa_pairs)}] Processing: {item_id} - {item['question']}")
            # Extract metadata fields
            meta = {k: v for k, v in item.items() if k not in ["question", "gold_answer"]}
            record = self.process_example(
                question=item["question"],
                gold_answer=item["gold_answer"],
                top_k=top_k,
                metadata=meta
            )
            labeled_records.append(record)
            completed_ids.add(item_id)
            _save_checkpoint(labeled_records)

        # Final checkpoint ensure
        _save_checkpoint(labeled_records)

        # Summary statistics
        total = len(labeled_records)
        label_1_count = sum(1 for r in labeled_records if r["label"] == 1)
        label_0_count = total - label_1_count
        direct_correct_count = sum(1 for r in labeled_records if r["direct_correct"])
        rag_correct_count = sum(1 for r in labeled_records if r["rag_correct"])

        # Retrieval hurts case: direct was correct but RAG became incorrect
        retrieval_hurt_count = sum(
            1 for r in labeled_records if r["direct_correct"] and not r["rag_correct"]
        )

        stats = {
            "total_examples": total,
            "label_0_count": label_0_count,
            "label_1_count": label_1_count,
            "direct_accuracy": round(direct_correct_count / total, 4) if total > 0 else 0.0,
            "rag_accuracy": round(rag_correct_count / total, 4) if total > 0 else 0.0,
            "retrieval_hurt_count": retrieval_hurt_count,
            "output_file": str(target_path),
            "records": labeled_records
        }

        return stats

