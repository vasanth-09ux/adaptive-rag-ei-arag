import sys
import io
import json
import time
from pathlib import Path
# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from src.classifier.model import SmallRetrievalClassifier

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    print("=" * 70)
    print("=== TRAINING EXPANDED COMPACT CLASSIFIER (1536 -> 32 -> 2) ===")
    print("=" * 70)

    set_seed(42)

    data_path = PROJECT_ROOT / "data" / "expanded_layer1_embeddings.pt"
    data = torch.load(data_path, map_location="cpu", weights_only=True)

    embeddings = data["embeddings"]
    labels = data["labels"]
    splits = data["splits"]
    ids = data["ids"]
    records = data["records"]

    # Filter splits
    train_mask = [s == "train" for s in splits]
    val_mask = [s == "validation" for s in splits]
    test_mask = [s == "test" for s in splits]

    X_train = embeddings[train_mask]
    y_train = labels[train_mask]

    X_val = embeddings[val_mask]
    y_val = labels[val_mask]

    X_test = embeddings[test_mask]
    y_test = labels[test_mask]
    test_ids = [ids[i] for i, m in enumerate(test_mask) if m]
    test_records = [records[i] for i, m in enumerate(test_mask) if m]

    print(f"Dataset summary:")
    print(f"  Train:      {X_train.shape[0]} samples (L0={sum(y_train==0).item()}, L1={sum(y_train==1).item()})")
    print(f"  Validation: {X_val.shape[0]} samples (L0={sum(y_val==0).item()}, L1={sum(y_val==1).item()})")
    print(f"  Test:       {X_test.shape[0]} samples (L0={sum(y_test==0).item()}, L1={sum(y_test==1).item()})")

    # Initialize model
    model = SmallRetrievalClassifier(input_dim=1536, hidden_dim=32, num_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel Architecture: 1536 -> 32 -> 2 (ReLU)")
    print(f"Total Parameters:   {total_params:,}")

    best_val_loss = float("inf")
    best_epoch = 0
    best_state_dict = None
    epoch_logs = []

    print("\n--- Training Epochs ---")
    for epoch in range(1, 51):
        # Train
        model.train()
        optimizer.zero_grad()
        logits = model(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

        # Evaluate Train
        model.eval()
        with torch.no_grad():
            train_preds = logits.argmax(dim=-1)
            train_acc = (train_preds == y_train).float().mean().item()

            # Validation
            val_logits = model(X_val)
            val_loss = criterion(val_logits, y_val).item()
            val_preds = val_logits.argmax(dim=-1)
            val_acc = (val_preds == y_val).float().mean().item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            checkpoint_mark = " *BEST*"
        else:
            checkpoint_mark = ""

        epoch_logs.append({
            "epoch": epoch,
            "train_loss": round(loss.item(), 5),
            "train_acc": round(train_acc * 100, 2),
            "val_loss": round(val_loss, 5),
            "val_acc": round(val_acc * 100, 2)
        })

        if epoch % 5 == 0 or epoch == 1 or checkpoint_mark:
            print(f"Epoch {epoch:02d}/50 | Train Loss: {loss.item():.4f} Acc: {train_acc*100:5.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc*100:5.2f}%{checkpoint_mark}")

    print(f"\nBest model selected at Epoch {best_epoch} with Val Loss = {best_val_loss:.5f}")

    # Load best checkpoint
    model.load_state_dict(best_state_dict)
    model.eval()

    # Save model checkpoint
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = models_dir / "classifier_expanded.pt"
    torch.save(best_state_dict, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")

    # Evaluate on Train, Validation, Test
    with torch.no_grad():
        train_logits = model(X_train)
        train_preds = train_logits.argmax(dim=-1).numpy()
        train_y_np = y_train.numpy()

        val_logits = model(X_val)
        val_preds = val_logits.argmax(dim=-1).numpy()
        val_y_np = y_val.numpy()

        test_logits = model(X_test)
        test_probs = torch.softmax(test_logits, dim=-1).numpy()
        test_preds = test_logits.argmax(dim=-1).numpy()
        test_y_np = y_test.numpy()

    # Metrics on untouched test set
    test_acc = accuracy_score(test_y_np, test_preds)
    test_prec_macro = precision_score(test_y_np, test_preds, average="macro", zero_division=0)
    test_rec_macro = recall_score(test_y_np, test_preds, average="macro", zero_division=0)
    test_f1_macro = f1_score(test_y_np, test_preds, average="macro", zero_division=0)

    # Class 1 metrics
    test_prec_c1 = precision_score(test_y_np, test_preds, pos_label=1, zero_division=0)
    test_rec_c1 = recall_score(test_y_np, test_preds, pos_label=1, zero_division=0)
    test_f1_c1 = f1_score(test_y_np, test_preds, pos_label=1, zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(test_y_np, test_preds)
    tn, fp, fn, tp = cm.ravel()

    # Majority baseline on test set
    maj_label = 0
    maj_acc = sum(test_y_np == maj_label) / len(test_y_np)

    pred_counts = Counter(test_preds.tolist())

    # Misclassified test items
    misclassified = []
    for idx, (tid, true_l, pred_l, prob, rec) in enumerate(zip(test_ids, test_y_np, test_preds, test_probs, test_records)):
        if true_l != pred_l:
            misclassified.append({
                "id": tid,
                "question": rec.get("question", ""),
                "gold_answer": rec.get("gold_answer", ""),
                "true_label": int(true_l),
                "pred_label": int(pred_l),
                "prob_l0": round(float(prob[0]), 4),
                "prob_l1": round(float(prob[1]), 4),
                "category": rec.get("category", "")
            })

    # Save metadata
    meta = {
        "model_architecture": "SmallRetrievalClassifier (1536 -> 32 -> 2)",
        "parameter_count": total_params,
        "training_samples": int(X_train.shape[0]),
        "validation_samples": int(X_val.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val_loss, 5),
        "test_metrics": {
            "accuracy": round(float(test_acc), 4),
            "macro_precision": round(float(test_prec_macro), 4),
            "macro_recall": round(float(test_rec_macro), 4),
            "macro_f1": round(float(test_f1_macro), 4),
            "label_1_precision": round(float(test_prec_c1), 4),
            "label_1_recall": round(float(test_rec_c1), 4),
            "label_1_f1": round(float(test_f1_c1), 4),
            "majority_baseline_acc": round(float(maj_acc), 4),
            "confusion_matrix": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp)
            },
            "predicted_distribution": {
                "label_0": int(pred_counts.get(0, 0)),
                "label_1": int(pred_counts.get(1, 0))
            }
        },
        "misclassified_test_ids": misclassified,
        "epoch_history": epoch_logs
    }

    meta_path = models_dir / "classifier_expanded_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"Saved metadata to {meta_path}")

    # Print Final Summary
    print("\n" + "=" * 70)
    print("=== FINAL CLASSIFIER EVALUATION REPORT ===")
    print("=" * 70)
    print(f"Parameters:              {total_params:,}")
    print(f"Training Samples:        {X_train.shape[0]} (Original: 70)")
    print(f"Validation Samples:      {X_val.shape[0]} (Original: 15)")
    print(f"Test Samples:            {X_test.shape[0]} (Original: 15)")
    print(f"Best Validation Epoch:   Epoch {best_epoch} (Val Loss: {best_val_loss:.5f})")
    print(f"Train Accuracy:          {accuracy_score(train_y_np, train_preds)*100:.2f}%")
    print(f"Validation Accuracy:     {accuracy_score(val_y_np, val_preds)*100:.2f}%")
    print(f"Test Accuracy:           {test_acc*100:.2f}% (Majority Baseline: {maj_acc*100:.2f}%)")
    print(f"Test Macro Precision:    {test_prec_macro*100:.2f}%")
    print(f"Test Macro Recall:       {test_rec_macro*100:.2f}%")
    print(f"Test Macro F1:           {test_f1_macro*100:.2f}%")
    print(f"Label-1 Precision:       {test_prec_c1*100:.2f}%")
    print(f"Label-1 Recall:          {test_rec_c1*100:.2f}%")
    print(f"Label-1 F1:              {test_f1_c1*100:.2f}%")
    print(f"Test Confusion Matrix:   TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"Test Predictions Dist:   Label 0 = {pred_counts.get(0,0)}, Label 1 = {pred_counts.get(1,0)}")
    print(f"Misclassified Count:     {len(misclassified)}/{len(test_y_np)}")

    if misclassified:
        print("\n--- Misclassified Test Queries ---")
        for m in misclassified:
            print(f"  [{m['id']}] True: {m['true_label']} | Pred: {m['pred_label']} (P0={m['prob_l0']:.2f}, P1={m['prob_l1']:.2f}) | {m['question']}")

if __name__ == "__main__":
    main()
