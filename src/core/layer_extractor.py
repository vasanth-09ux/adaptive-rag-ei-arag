"""
EI-ARAG Layer-1 Representation Extractor.
Extracts the first contextualized transformer layer representations (hidden_states[1])
from raw input queries and computes attention-mask-aware mean pooling.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import torch

from src.core.llm_loader import LLMManager
from config.config import config, DATA_DIR, AppConfig


class Layer1Extractor:
    """
    Extracts Layer-1 contextual embeddings from Qwen2.5 for adaptive retrieval classification.
    Following Huang et al. (COLING 2025), Layer 1 provides discriminative contextualized
    signals regarding the model's intrinsic knowledge with minimal latency.
    """

    def __init__(
        self,
        app_config: AppConfig = config,
        llm_manager: Optional[LLMManager] = None,
        target_layer: int = 1
    ):
        self.config = app_config
        self.llm_manager = llm_manager or LLMManager.get_instance(cfg=self.config.llm)
        self.model = self.llm_manager.model
        self.tokenizer = self.llm_manager.tokenizer
        self.device = self.llm_manager.device
        self.target_layer = target_layer

        # Derive hidden dimension dynamically from model config
        self.hidden_size = getattr(self.model.config, "hidden_size", 1536)

        # Set pad token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _mean_pool(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Computes attention-mask-aware mean pooling across the token sequence dimension.
        hidden_states: (batch_size, seq_len, hidden_size)
        attention_mask: (batch_size, seq_len)
        Returns: (batch_size, hidden_size) on CPU in float32.
        """
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * input_mask_expanded, dim=1)
        sum_mask = input_mask_expanded.sum(dim=1).clamp(min=1e-9)
        pooled = sum_embeddings / sum_mask
        return pooled.detach().cpu().to(torch.float32)

    def extract_single(self, text: str) -> torch.Tensor:
        """
        Extracts Layer-1 mean-pooled representation for a single raw question string.
        Returns 1D tensor of shape (hidden_size,).
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)

        with torch.inference_mode():
            outputs = self.model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                output_hidden_states=True
            )
            # outputs.hidden_states[0] = embedding lookup
            # outputs.hidden_states[1] = 1st transformer layer output
            layer_hidden = outputs.hidden_states[self.target_layer]
            pooled = self._mean_pool(layer_hidden, inputs.attention_mask)

        return pooled.squeeze(0)  # Shape: (hidden_size,)

    def extract_batch(self, texts: List[str], batch_size: int = 16) -> torch.Tensor:
        """
        Extracts Layer-1 representations for a list of raw question strings in batches.
        Returns 2D tensor of shape (len(texts), hidden_size).
        """
        all_embeddings: List[torch.Tensor] = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)

            with torch.inference_mode():
                outputs = self.model(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    output_hidden_states=True
                )
                layer_hidden = outputs.hidden_states[self.target_layer]
                pooled = self._mean_pool(layer_hidden, inputs.attention_mask)
                all_embeddings.append(pooled)

        return torch.cat(all_embeddings, dim=0)  # Shape: (N, hidden_size)

    def extract_dataset(
        self,
        dataset_path: Optional[Path] = None,
        output_path: Optional[Path] = None,
        batch_size: int = 16
    ) -> Dict[str, Any]:
        """
        Extracts Layer-1 representations for all questions in the labeled dataset JSON.
        Saves representations and full audit metadata to a .pt file.
        """
        in_path = Path(dataset_path or (DATA_DIR / "labeled_dataset.json"))
        out_path = Path(output_path or (DATA_DIR / "layer1_embeddings.pt"))
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(in_path, "r", encoding="utf-8") as f:
            records = json.load(f)

        print(f"Extracting Layer-{self.target_layer} representations for {len(records)} questions...")
        questions = [r["question"] for r in records]

        # Extract embeddings in batches
        embeddings = self.extract_batch(questions, batch_size=batch_size)

        # Build structured dataset
        dataset_records = []
        for r, emb in zip(records, embeddings):
            dataset_records.append({
                "id": r["id"],
                "question": r["question"],
                "gold_answer": r.get("gold_answer"),
                "label": r["label"],
                "split": r["split"],
                "category": r["category"],
                "source": r.get("source"),
                "embedding": emb  # 1D CPU tensor
            })

        data_dict = {
            "embeddings": embeddings,  # (N, hidden_size) float32 CPU tensor
            "labels": torch.tensor([r["label"] for r in records], dtype=torch.long),
            "ids": [r["id"] for r in records],
            "splits": [r["split"] for r in records],
            "categories": [r["category"] for r in records],
            "records": dataset_records,
            "metadata": {
                "num_examples": len(records),
                "hidden_size": self.hidden_size,
                "layer_index": self.target_layer,
                "pooling_method": "attention_mask_aware_mean_pooling",
                "model_id": self.config.llm.model_id,
                "device_extracted": str(self.device),
                "device_stored": "cpu",
                "dtype": "torch.float32"
            }
        }

        torch.save(data_dict, out_path)
        print(f"Saved {len(records)} representations to {out_path} ({out_path.stat().st_size / 1024:.1f} KB).")

        return data_dict
