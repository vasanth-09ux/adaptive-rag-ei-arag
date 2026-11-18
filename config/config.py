"""
Global Configuration for Baseline RAG and EI-ARAG Adaptive Pipeline.
"""

from pathlib import Path
from dataclasses import dataclass, field
import torch

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_CORPUS_DIR = DATA_DIR / "raw_corpus"
MODELS_DIR = PROJECT_ROOT / "models"

@dataclass
class LLMConfig:
    """LLM model configuration."""
    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    max_new_tokens: int = 128
    temperature: float = 0.0
    # Target transformer layer for EI-ARAG embedding extraction (1st contextualized layer)
    target_layer_idx: int = 1

@dataclass
class ClassifierConfig:
    """CPU Binary Classifier configuration for EI-ARAG."""
    # hidden_dim is resolved dynamically from model.config.hidden_size at runtime if None
    input_dim: int | None = None
    hidden_dim: int = 32
    dropout_rate: float = 0.0
    threshold: float = 0.5
    device: str = "cpu"
    model_save_path: Path = MODELS_DIR / "classifier_expanded.pt"

@dataclass
class RetrievalConfig:
    """Vector database and retrieval configuration."""
    embedding_model_id: str = "BAAI/bge-small-en-v1.5"
    retriever_type: str = "faiss"
    top_k: int = 3
    chunk_size: int = 256
    chunk_overlap: int = 32
    device: str = "cpu"
    index_path: Path = DATA_DIR / "faiss_index.bin"
    metadata_path: Path = DATA_DIR / "chunks_metadata.json"

@dataclass
class AppConfig:
    """Top-level aggregated configuration."""
    llm: LLMConfig = field(default_factory=LLMConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)

# Default configuration instance
config = AppConfig()
