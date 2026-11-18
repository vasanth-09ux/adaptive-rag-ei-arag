import sys
import io
import time
from pathlib import Path
# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from src.core.layer_extractor import Layer1Extractor

def main():
    print("=== Extracting Layer-1 Representations for Expanded Dataset (310 samples) ===")
    dataset_path = PROJECT_ROOT / "data" / "expanded_labeled_dataset.json"
    output_path = PROJECT_ROOT / "data" / "expanded_layer1_embeddings.pt"

    extractor = Layer1Extractor()
    start_time = time.perf_counter()
    stats = extractor.extract_dataset(
        dataset_path=dataset_path,
        output_path=output_path,
        batch_size=16
    )
    elapsed = time.perf_counter() - start_time

    print(f"\nExtraction completed in {elapsed:.2f}s.")
    print(f"Shape: {stats['tensor_shape']}")
    print(f"Total Samples: {stats['num_samples']}")
    print(f"Splits: {stats['split_counts']}")
    print(f"Labels: {stats['label_counts']}")
    print(f"Saved to {output_path} ({output_path.stat().st_size / (1024*1024):.2f} MB).")

    # Double check tensor validity
    data = torch.load(output_path, map_location="cpu", weights_only=True)
    embeddings = data["embeddings"]
    has_nan = torch.isnan(embeddings).any().item()
    has_inf = torch.isinf(embeddings).any().item()
    print(f"Verification: NaN detected = {has_nan}, Inf detected = {has_inf}")
    print(f"Embedding norm mean: {embeddings.norm(dim=-1).mean().item():.4f}")

if __name__ == "__main__":
    main()
