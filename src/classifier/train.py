"""
EI-ARAG Binary Classifier Training and Evaluation Pipeline.
Trains a lightweight MLP on Layer-1 Qwen representations using CPU,
with early stopping on the validation split and final evaluation on the untouched test split.
"""

import sys
import io
import json
import random
import copy
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from src.classifier.model import RetrievalClassifier
from config.config import DATA_DIR, MODELS_DIR


def set_seed(seed: int = 42) -> None:
    """Sets deterministic random seeds for full reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calculate_metrics(y_true: torch.Tensor, y_pred: torch.Tensor) -> Dict[str, Any]:
    """
    Computes accuracy, precision (class 1), recall (class 1), F1 (class 1),
    and confusion matrix [[TN, FP], [FN, TP]].
    """
    total = len(y_true)
    if total == 0:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "confusion_matrix": [[0, 0], [0, 0]]}

    accuracy = (y_true == y_pred).float().mean().item()

    tp = ((y_pred == 1) & (y_true == 1)).sum().item()
    fp = ((y_pred == 1) & (y_true == 0)).sum().item()
    fn = ((y_pred == 0) & (y_true == 1)).sum().item()
    tn = ((y_pred == 0) & (y_true == 0)).sum().item()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "counts": {"total": total, "tp": tp, "fp": fp, "fn": fn, "tn": tn}
    }


class ClassifierTrainer:
    """Manages training, validation-based checkpointing, and evaluation of the retrieval classifier."""

    def __init__(
        self,
        data_path: Optional[Path] = None,
        save_dir: Optional[Path] = None,
        hidden_dim1: int = 128,
        hidden_dim2: int = 64,
        learning_rate: float = 1e-3,
        max_epochs: int = 50,
        batch_size: int = 16,
        seed: int = 42
    ):
        self.data_path = Path(data_path or (DATA_DIR / "layer1_embeddings.pt"))
        self.save_dir = Path(save_dir or MODELS_DIR)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.seed = seed

        set_seed(self.seed)

    def load_data(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[str]]]:
        """Loads and organizes representations by train, validation, and test splits."""
        data = torch.load(self.data_path, weights_only=False)
        embeddings = data["embeddings"]
        labels = data["labels"]
        ids = data["ids"]
        splits = data["splits"]

        split_tensors = {}
        split_ids = {}

        for s in ["train", "validation", "test"]:
            mask = [idx for idx, sp in enumerate(splits) if sp == s]
            split_tensors[f"X_{s}"] = embeddings[mask]
            split_tensors[f"y_{s}"] = labels[mask]
            split_ids[s] = [ids[idx] for idx in mask]

        return split_tensors, split_ids

    def train(self) -> Dict[str, Any]:
        """Executes the training loop with early stopping based on validation loss."""
        split_tensors, split_ids = self.load_data()

        X_train, y_train = split_tensors["X_train"], split_tensors["y_train"]
        X_val, y_val = split_tensors["X_validation"], split_tensors["y_validation"]
        X_test, y_test = split_tensors["X_test"], split_tensors["y_test"]

        input_dim = X_train.shape[1]

        # Initialize model
        model = RetrievalClassifier(
            input_dim=input_dim,
            hidden_dim1=self.hidden_dim1,
            hidden_dim2=self.hidden_dim2,
            num_classes=2,
            dropout_rate=0.0
        )
        total_params = model.count_parameters()

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed)
        )

        best_val_loss = float("inf")
        best_val_acc = 0.0
        best_epoch = 0
        best_model_state = copy.deepcopy(model.state_dict())

        training_history: List[Dict[str, Any]] = []

        print(f"Training RetrievalClassifier ({total_params:,} parameters) on CPU...")
        print(f"Dataset splits: Train={len(y_train)}, Validation={len(y_val)}, Test={len(y_test)}")
        print(f"Epochs: {self.max_epochs} | Batch Size: {self.batch_size} | Learning Rate: {self.learning_rate}\n")

        for epoch in range(1, self.max_epochs + 1):
            # --- Training Pass ---
            model.train()
            total_train_loss = 0.0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item() * len(batch_y)

            train_loss = total_train_loss / len(y_train)

            # --- Evaluation Passes (Train & Validation) ---
            model.eval()
            with torch.no_grad():
                train_logits = model(X_train)
                train_preds = torch.argmax(train_logits, dim=-1)
                train_acc = (train_preds == y_train).float().mean().item()

                val_logits = model(X_val)
                val_loss = criterion(val_logits, y_val).item()
                val_preds = torch.argmax(val_logits, dim=-1)
                val_acc = (val_preds == y_val).float().mean().item()

            training_history.append({
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "train_acc": round(train_acc, 4),
                "val_loss": round(val_loss, 4),
                "val_acc": round(val_acc, 4)
            })

            # Checkpoint on best validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_epoch = epoch
                best_model_state = copy.deepcopy(model.state_dict())

        # Load best model for evaluation
        model.load_state_dict(best_model_state)
        model.eval()

        # Save best model checkpoint
        model_save_path = self.save_dir / "classifier.pt"
        torch.save(best_model_state, model_save_path)

        # --- Comprehensive Metrics Computation ---
        with torch.no_grad():
            # 1. Train Evaluation
            train_preds = model.predict(X_train)
            train_metrics = calculate_metrics(y_train, train_preds)

            # 2. Validation Evaluation
            val_preds = model.predict(X_val)
            val_metrics = calculate_metrics(y_val, val_preds)

            # 3. Final Test Evaluation (Untouched split)
            test_preds = model.predict(X_test)
            test_metrics = calculate_metrics(y_test, test_preds)

        # Majority Class Baseline on Test Split (Always Predict 0)
        majority_preds = torch.zeros_like(y_test)
        majority_acc = (majority_preds == y_test).float().mean().item()

        # Test misclassified IDs
        misclassified_mask = (test_preds != y_test)
        misclassified_test_ids = [
            {
                "id": split_ids["test"][i],
                "actual_label": int(y_test[i].item()),
                "predicted_label": int(test_preds[i].item())
            }
            for i in range(len(y_test))
            if misclassified_mask[i]
        ]

        # Overfitting Diagnosis
        overfitting_gap = train_metrics["accuracy"] - test_metrics["accuracy"]

        summary = {
            "metadata": {
                "architecture": "MLP (Linear 1536->128 -> ReLU -> Linear 128->64 -> ReLU -> Linear 64->2)",
                "input_dim": input_dim,
                "hidden_dim1": self.hidden_dim1,
                "hidden_dim2": self.hidden_dim2,
                "total_parameters": total_params,
                "optimizer": "Adam",
                "learning_rate": self.learning_rate,
                "max_epochs": self.max_epochs,
                "batch_size": self.batch_size,
                "seed": self.seed,
                "best_epoch": best_epoch,
                "best_val_loss": round(best_val_loss, 4),
                "best_val_acc": round(best_val_acc, 4),
                "model_save_path": str(model_save_path)
            },
            "split_sizes": {
                "train": len(y_train),
                "validation": len(y_val),
                "test": len(y_test)
            },
            "train_metrics": train_metrics,
            "validation_metrics": val_metrics,
            "test_metrics": test_metrics,
            "test_label_distribution": {
                "actual": {"label_0": int((y_test == 0).sum().item()), "label_1": int((y_test == 1).sum().item())},
                "predicted": {"label_0": int((test_preds == 0).sum().item()), "label_1": int((test_preds == 1).sum().item())}
            },
            "majority_baseline_test_accuracy": round(majority_acc, 4),
            "misclassified_test_examples": misclassified_test_ids,
            "overfitting_analysis": {
                "train_accuracy": train_metrics["accuracy"],
                "validation_accuracy": val_metrics["accuracy"],
                "test_accuracy": test_metrics["accuracy"],
                "generalization_gap": round(overfitting_gap, 4),
                "status": "Moderate Overfitting" if overfitting_gap > 0.15 else "Well Generalizing"
            },
            "training_history": training_history
        }

        # Save metadata to JSON
        metadata_save_path = self.save_dir / "classifier_metadata.json"
        with open(metadata_save_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        return summary


def main():
    print("=== Training EI-ARAG Binary Classifier ===")
    trainer = ClassifierTrainer(
        hidden_dim1=128,
        hidden_dim2=64,
        learning_rate=1e-3,
        max_epochs=50,
        batch_size=16,
        seed=42
    )
    results = trainer.train()
    meta = results["metadata"]
    train_m = results["train_metrics"]
    val_m = results["validation_metrics"]
    test_m = results["test_metrics"]
    maj_acc = results["majority_baseline_test_accuracy"]

    print("\n" + "="*60)
    print("=== TRAINING RESULTS & METRICS ===")
    print("="*60)
    print(f"Model Architecture:    {meta['architecture']}")
    print(f"Trainable Parameters:  {meta['total_parameters']:,}")
    print(f"Best Checkpoint Epoch: {meta['best_epoch']} / {meta['max_epochs']}")
    print(f"Best Validation Loss:  {meta['best_val_loss']}")

    print("\n" + "-"*60)
    print(f"TRAIN SPLIT (N={results['split_sizes']['train']}):")
    print(f"  Accuracy:  {train_m['accuracy'] * 100:.2f}%")
    print(f"  Precision: {train_m['precision'] * 100:.2f}%")
    print(f"  Recall:    {train_m['recall'] * 100:.2f}%")
    print(f"  F1 Score:  {train_m['f1'] * 100:.2f}%")
    print(f"  Confusion Matrix: [[TN={train_m['confusion_matrix'][0][0]}, FP={train_m['confusion_matrix'][0][1]}], [FN={train_m['confusion_matrix'][1][0]}, TP={train_m['confusion_matrix'][1][1]}]]")

    print("\n" + "-"*60)
    print(f"VALIDATION SPLIT (N={results['split_sizes']['validation']}) [Model Selection]:")
    print(f"  Accuracy:  {val_m['accuracy'] * 100:.2f}%")
    print(f"  Precision: {val_m['precision'] * 100:.2f}%")
    print(f"  Recall:    {val_m['recall'] * 100:.2f}%")
    print(f"  F1 Score:  {val_m['f1'] * 100:.2f}%")
    print(f"  Confusion Matrix: [[TN={val_m['confusion_matrix'][0][0]}, FP={val_m['confusion_matrix'][0][1]}], [FN={val_m['confusion_matrix'][1][0]}, TP={val_m['confusion_matrix'][1][1]}]]")

    print("\n" + "="*60)
    print(f"TEST SPLIT (N={results['split_sizes']['test']}) [Untouched Final Evaluation]:")
    print("="*60)
    print(f"  Accuracy:  {test_m['accuracy'] * 100:.2f}%")
    print(f"  Precision: {test_m['precision'] * 100:.2f}%")
    print(f"  Recall:    {test_m['recall'] * 100:.2f}%")
    print(f"  F1 Score:  {test_m['f1'] * 100:.2f}%")
    print(f"  Confusion Matrix: [[TN={test_m['confusion_matrix'][0][0]}, FP={test_m['confusion_matrix'][0][1]}], [FN={test_m['confusion_matrix'][1][0]}, TP={test_m['confusion_matrix'][1][1]}]]")

    print("\n" + "-"*60)
    print("=== BASELINE COMPARISON & ERROR ANALYSIS ===")
    print("-"*60)
    print(f"Majority-Class Baseline Accuracy (Always 0): {maj_acc * 100:.2f}%")
    print(f"EI-ARAG Classifier Test Accuracy:           {test_m['accuracy'] * 100:.2f}%")
    delta = test_m['accuracy'] - maj_acc
    print(f"Performance Delta over Majority Baseline:   {'+' if delta >= 0 else ''}{delta * 100:.2f}%")

    print(f"\nActual Test Label Distribution:    {results['test_label_distribution']['actual']}")
    print(f"Predicted Test Label Distribution: {results['test_label_distribution']['predicted']}")
    print(f"Misclassified Test IDs ({len(results['misclassified_test_examples'])}):")
    for item in results["misclassified_test_examples"]:
        print(f"  - {item['id']}: Actual Label={item['actual_label']}, Predicted Label={item['predicted_label']}")

    print(f"\nOverfitting Status: {results['overfitting_analysis']['status']} (Train Acc: {train_m['accuracy']*100:.1f}%, Test Acc: {test_m['accuracy']*100:.1f}%, Gap: {results['overfitting_analysis']['generalization_gap']*100:.1f}%)")
    print(f"Saved model to:    {meta['model_save_path']}")
    print(f"Saved metadata to: {MODELS_DIR / 'classifier_metadata.json'}")

if __name__ == "__main__":
    main()
