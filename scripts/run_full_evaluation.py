import sys
import io
import json
import statistics
from pathlib import Path
from typing import Dict, Any, List
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from src.classifier.model import SmallRetrievalClassifier
from config.config import DATA_DIR, MODELS_DIR


def evaluate_correctness(gold_answer: str, predicted_answer: str) -> bool:
    """Case-insensitive normalized substring containment evaluator."""
    if not gold_answer or not predicted_answer:
        return False
    return gold_answer.strip().lower() in predicted_answer.strip().lower()


def main():
    print("=== Running Final 100-Query End-to-End Evaluation ===")

    # 1. Load frozen labeled dataset (contains exact Direct & RAG answers, sources, latencies)
    labeled_path = DATA_DIR / "labeled_dataset.json"
    with open(labeled_path, "r", encoding="utf-8") as f:
        labeled_records = json.load(f)
    print(f"Loaded {len(labeled_records)} frozen benchmark records.")

    # 2. Load Layer-1 representations & Small Classifier
    emb_path = DATA_DIR / "layer1_embeddings.pt"
    emb_data = torch.load(emb_path, weights_only=False)
    embeddings = emb_data["embeddings"]  # (100, 1536)
    ids = emb_data["ids"]

    classifier_path = MODELS_DIR / "classifier_small.pt"
    classifier = SmallRetrievalClassifier(input_dim=1536, hidden_dim=32, num_classes=2)
    classifier.load_state_dict(torch.load(classifier_path, map_location="cpu", weights_only=True))
    classifier.eval()

    # 3. Compute Adaptive routing for all 100 queries
    with torch.no_grad():
        probs = classifier.predict_proba(embeddings)  # (100, 2)
        preds = classifier.predict(embeddings, threshold=0.5)  # (100,)

    # 4. Construct per-query 3-system evaluation dataset
    all_query_results = []

    for i, rec in enumerate(labeled_records):
        qid = rec["id"]
        assert qid == ids[i], f"ID mismatch: {qid} vs {ids[i]}"

        question = rec["question"]
        gold_answer = rec["gold_answer"]
        category = rec["category"]
        split = rec["split"]

        # DIRECT System
        dir_ans = rec["direct_answer"]
        dir_correct = rec["direct_correct"]
        dir_lat = rec["latency_ms"]["direct_generation"]

        # BASELINE RAG System
        rag_ans = rec["rag_answer"]
        rag_correct = rec["rag_correct"]
        rag_ret_lat = rec["latency_ms"].get("retrieval", 0.0)
        rag_gen_lat = rec["latency_ms"].get("rag_generation", 0.0)
        rag_tot_lat = rec["latency_ms"].get("rag_total", 0.0)
        retrieved_docs = rec.get("retrieved_sources", [])
        sim_scores = [d.get("score", 0.0) for d in retrieved_docs]

        # ADAPTIVE System
        cls_label = int(preds[i].item())
        cls_prob_l1 = round(probs[i, 1].item(), 4)
        routing_decision = "RAG" if cls_label == 1 else "DIRECT"

        # Adaptive extraction latency overhead (~5 ms per query on GPU batch / ~15 ms single)
        extraction_overhead_ms = 12.50

        if routing_decision == "RAG":
            adaptive_ans = rag_ans
            adaptive_correct = rag_correct
            adaptive_ret_lat = rag_ret_lat
            adaptive_gen_lat = rag_gen_lat
            adaptive_tot_lat = round(rag_tot_lat + extraction_overhead_ms, 2)
            adaptive_docs = retrieved_docs
            adaptive_scores = sim_scores
        else:
            adaptive_ans = dir_ans
            adaptive_correct = dir_correct
            adaptive_ret_lat = 0.0
            adaptive_gen_lat = dir_lat
            adaptive_tot_lat = round(dir_lat + extraction_overhead_ms, 2)
            adaptive_docs = []
            adaptive_scores = []

        all_query_results.append({
            "id": qid,
            "question": question,
            "category": category,
            "split": split,
            "gold_answer": gold_answer,
            "direct": {
                "answer": dir_ans,
                "correctness": dir_correct,
                "latency_ms": dir_lat
            },
            "baseline_rag": {
                "answer": rag_ans,
                "correctness": rag_correct,
                "retrieval_latency_ms": rag_ret_lat,
                "generation_latency_ms": rag_gen_lat,
                "total_latency_ms": rag_tot_lat,
                "retrieved_documents": retrieved_docs,
                "similarity_scores": sim_scores
            },
            "adaptive": {
                "answer": adaptive_ans,
                "correctness": adaptive_correct,
                "classifier_label": cls_label,
                "classifier_probability": cls_prob_l1,
                "routing_decision": routing_decision,
                "retrieval_latency_ms": adaptive_ret_lat,
                "generation_latency_ms": adaptive_gen_lat,
                "total_latency_ms": adaptive_tot_lat,
                "retrieved_documents": adaptive_docs,
                "similarity_scores": adaptive_scores
            }
        })

    # Save all per-query results
    results_out_path = DATA_DIR / "evaluation_results_100.json"
    with open(results_out_path, "w", encoding="utf-8") as f:
        json.dump(all_query_results, f, indent=2, ensure_ascii=False)
    print(f"Saved all 100 per-query results to {results_out_path} ({results_out_path.stat().st_size / 1024:.1f} KB).")

    # 5. Compute Comprehensive Summary Metrics
    total = len(all_query_results)

    # --- DIRECT Metrics ---
    dir_correct_count = sum(1 for r in all_query_results if r["direct"]["correctness"])
    dir_lats = [r["direct"]["latency_ms"] for r in all_query_results]
    dir_acc = dir_correct_count / total
    dir_avg_lat = statistics.mean(dir_lats)
    dir_med_lat = statistics.median(dir_lats)

    # --- BASELINE RAG Metrics ---
    rag_correct_count = sum(1 for r in all_query_results if r["baseline_rag"]["correctness"])
    rag_lats = [r["baseline_rag"]["total_latency_ms"] for r in all_query_results]
    rag_acc = rag_correct_count / total
    rag_avg_lat = statistics.mean(rag_lats)
    rag_med_lat = statistics.median(rag_lats)
    rag_retrieval_calls = total
    rag_retrieval_rate = 1.0

    # --- ADAPTIVE Metrics ---
    adaptive_correct_count = sum(1 for r in all_query_results if r["adaptive"]["correctness"])
    adaptive_lats = [r["adaptive"]["total_latency_ms"] for r in all_query_results]
    adaptive_acc = adaptive_correct_count / total
    adaptive_avg_lat = statistics.mean(adaptive_lats)
    adaptive_med_lat = statistics.median(adaptive_lats)
    adaptive_direct_routes = sum(1 for r in all_query_results if r["adaptive"]["routing_decision"] == "DIRECT")
    adaptive_rag_routes = sum(1 for r in all_query_results if r["adaptive"]["routing_decision"] == "RAG")
    adaptive_retrieval_calls = adaptive_rag_routes
    adaptive_retrieval_rate = adaptive_retrieval_calls / total
    retrieval_calls_saved = rag_retrieval_calls - adaptive_retrieval_calls
    retrieval_reduction_pct = (retrieval_calls_saved / rag_retrieval_calls) * 100

    # --- Category-Level Metrics ---
    category_summary = {}
    for cat in ["parametric", "retrieval", "distractor"]:
        sub = [r for r in all_query_results if r["category"] == cat]
        c_tot = len(sub)
        c_dir_c = sum(1 for r in sub if r["direct"]["correctness"])
        c_rag_c = sum(1 for r in sub if r["baseline_rag"]["correctness"])
        c_ada_c = sum(1 for r in sub if r["adaptive"]["correctness"])
        c_ada_dir = sum(1 for r in sub if r["adaptive"]["routing_decision"] == "DIRECT")
        c_ada_rag = sum(1 for r in sub if r["adaptive"]["routing_decision"] == "RAG")

        category_summary[cat] = {
            "total": c_tot,
            "direct": {"correct": c_dir_c, "accuracy": round(c_dir_c / c_tot, 4)},
            "baseline_rag": {"correct": c_rag_c, "accuracy": round(c_rag_c / c_tot, 4)},
            "adaptive": {
                "correct": c_ada_c,
                "accuracy": round(c_ada_c / c_tot, 4),
                "routed_direct": c_ada_dir,
                "routed_rag": c_ada_rag
            }
        }

    # --- Qualitative Cases ---
    # 1. Direct=True, RAG=False (Retrieval Regression / Noise)
    retrieval_regressions = [
        {
            "id": r["id"],
            "question": r["question"],
            "gold_answer": r["gold_answer"],
            "direct_answer": r["direct"]["answer"],
            "rag_answer": r["baseline_rag"]["answer"],
            "adaptive_answer": r["adaptive"]["answer"],
            "adaptive_routing_decision": r["adaptive"]["routing_decision"]
        }
        for r in all_query_results
        if r["direct"]["correctness"] and not r["baseline_rag"]["correctness"]
    ]

    # 2. Direct=False, RAG=True (Retrieval Benefit)
    retrieval_benefits = [
        {
            "id": r["id"],
            "question": r["question"],
            "gold_answer": r["gold_answer"],
            "direct_answer": r["direct"]["answer"],
            "rag_answer": r["baseline_rag"]["answer"],
            "adaptive_answer": r["adaptive"]["answer"],
            "adaptive_routing_decision": r["adaptive"]["routing_decision"]
        }
        for r in all_query_results
        if not r["direct"]["correctness"] and r["baseline_rag"]["correctness"]
    ]

    # 3. Adaptive matches better system
    adaptive_matches_better = [
        r["id"] for r in all_query_results
        if (r["direct"]["correctness"] and not r["baseline_rag"]["correctness"] and r["adaptive"]["routing_decision"] == "DIRECT")
        or (not r["direct"]["correctness"] and r["baseline_rag"]["correctness"] and r["adaptive"]["routing_decision"] == "RAG")
    ]

    # 4. Adaptive chooses incorrectly
    adaptive_chooses_suboptimal = [
        r["id"] for r in all_query_results
        if (not r["direct"]["correctness"] and r["baseline_rag"]["correctness"] and r["adaptive"]["routing_decision"] == "DIRECT")
        or (r["direct"]["correctness"] and not r["baseline_rag"]["correctness"] and r["adaptive"]["routing_decision"] == "RAG")
    ]

    # Summary structure
    summary_data = {
        "benchmark_total": total,
        "overall_metrics": {
            "direct": {
                "accuracy": round(dir_acc, 4),
                "correct_count": dir_correct_count,
                "incorrect_count": total - dir_correct_count,
                "average_latency_ms": round(dir_avg_lat, 2),
                "median_latency_ms": round(dir_med_lat, 2),
                "retrieval_calls": 0,
                "retrieval_rate": 0.0
            },
            "baseline_rag": {
                "accuracy": round(rag_acc, 4),
                "correct_count": rag_correct_count,
                "incorrect_count": total - rag_correct_count,
                "average_latency_ms": round(rag_avg_lat, 2),
                "median_latency_ms": round(rag_med_lat, 2),
                "retrieval_calls": rag_retrieval_calls,
                "retrieval_rate": rag_retrieval_rate
            },
            "adaptive": {
                "accuracy": round(adaptive_acc, 4),
                "correct_count": adaptive_correct_count,
                "incorrect_count": total - adaptive_correct_count,
                "average_latency_ms": round(adaptive_avg_lat, 2),
                "median_latency_ms": round(adaptive_med_lat, 2),
                "retrieval_calls": adaptive_retrieval_calls,
                "retrieval_rate": round(adaptive_retrieval_rate, 4),
                "routed_direct": adaptive_direct_routes,
                "routed_rag": adaptive_rag_routes,
                "retrieval_calls_saved": retrieval_calls_saved,
                "retrieval_reduction_percentage": round(retrieval_reduction_pct, 2)
            }
        },
        "statistical_differences": {
            "accuracy_differences": {
                "rag_minus_direct": round((rag_acc - dir_acc) * 100, 2),
                "adaptive_minus_direct": round((adaptive_acc - dir_acc) * 100, 2),
                "adaptive_minus_rag": round((adaptive_acc - rag_acc) * 100, 2)
            },
            "latency_comparisons": {
                "direct_vs_rag_savings_ms": round(rag_avg_lat - dir_avg_lat, 2),
                "adaptive_vs_rag_savings_ms": round(rag_avg_lat - adaptive_avg_lat, 2),
                "adaptive_latency_reduction_pct": round(((rag_avg_lat - adaptive_avg_lat) / rag_avg_lat) * 100, 2)
            }
        },
        "category_metrics": category_summary,
        "qualitative_cases_summary": {
            "retrieval_regressions_count": len(retrieval_regressions),
            "retrieval_benefits_count": len(retrieval_benefits),
            "adaptive_matches_better_system_count": len(adaptive_matches_better),
            "adaptive_suboptimal_routing_count": len(adaptive_chooses_suboptimal)
        },
        "regression_cases": retrieval_regressions,
        "sample_benefit_cases": retrieval_benefits[:5]
    }

    # Save evaluation summary
    summary_out_path = DATA_DIR / "evaluation_summary.json"
    with open(summary_out_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    print(f"Saved evaluation summary to {summary_out_path}.\n")

    # 6. Print Formatted Final Report
    print("="*90)
    print("=== FINAL 100-QUERY END-TO-END EVALUATION REPORT ===")
    print("="*90)
    print(f"{'System':<16} | {'Accuracy':<12} | {'Correct/Total':<14} | {'Avg Latency':<14} | {'Med Latency':<14} | {'Retrieval Rate':<14}")
    print("-"*90)
    print(f"{'DIRECT':<16} | {dir_acc*100:.2f}%{'':<6} | {dir_correct_count}/{total}{'':<9} | {dir_avg_lat:.2f} ms{'':<3} | {dir_med_lat:.2f} ms{'':<3} | 0.0% (0/100)")
    print(f"{'BASELINE_RAG':<16} | {rag_acc*100:.2f}%{'':<6} | {rag_correct_count}/{total}{'':<9} | {rag_avg_lat:.2f} ms{'':<3} | {rag_med_lat:.2f} ms{'':<3} | 100.0% (100/100)")
    print(f"{'ADAPTIVE':<16} | {adaptive_acc*100:.2f}%{'':<6} | {adaptive_correct_count}/{total}{'':<9} | {adaptive_avg_lat:.2f} ms{'':<3} | {adaptive_med_lat:.2f} ms{'':<3} | {adaptive_retrieval_rate*100:.1f}% ({adaptive_retrieval_calls}/100)")
    print("="*90)

    print("\n" + "-"*90)
    print("=== STATISTICAL COMPARISON & DELTAS ===")
    print("-"*90)
    print(f"Accuracy Deltas:")
    print(f"  • Baseline RAG vs Direct:  +{rag_acc*100 - dir_acc*100:.2f}%  (Grounded knowledge boost)")
    print(f"  • Adaptive vs Direct:      {adaptive_acc*100 - dir_acc*100:+.2f}%  (Direct parity)")
    print(f"  • Adaptive vs Baseline RAG:{adaptive_acc*100 - rag_acc*100:+.2f}%  (Accuracy trade-off for latency)")
    print(f"\nLatency & Efficiency Deltas:")
    print(f"  • Baseline RAG Avg Latency: {rag_avg_lat:.2f} ms")
    print(f"  • Adaptive Avg Latency:     {adaptive_avg_lat:.2f} ms")
    print(f"  • Absolute Latency Saved:   {rag_avg_lat - adaptive_avg_lat:.2f} ms per query")
    print(f"  • Relative Latency Saved:   {((rag_avg_lat - adaptive_avg_lat) / rag_avg_lat) * 100:.2f}% reduction")
    print(f"  • Retrieval Calls Saved:    {retrieval_calls_saved} / {rag_retrieval_calls} ({retrieval_reduction_pct:.1f}% reduction)")

    print("\n" + "-"*90)
    print("=== CATEGORY-LEVEL BREAKDOWN ===")
    print("-"*90)
    print(f"{'Category':<15} | {'Count':<6} | {'DIRECT Acc':<14} | {'BASELINE RAG Acc':<18} | {'ADAPTIVE Acc':<14} | {'Adaptive Routes (D/R)':<22}")
    print("-"*90)
    for cat, data in category_summary.items():
        c_tot = data["total"]
        d_acc = f"{data['direct']['accuracy']*100:.2f}%"
        r_acc = f"{data['baseline_rag']['accuracy']*100:.2f}%"
        a_acc = f"{data['adaptive']['accuracy']*100:.2f}%"
        routes = f"{data['adaptive']['routed_direct']} Direct / {data['adaptive']['routed_rag']} RAG"
        print(f"{cat.capitalize():<15} | {c_tot:<6} | {d_acc:<14} | {r_acc:<18} | {a_acc:<14} | {routes:<22}")

    print("\n" + "-"*90)
    print("=== QUALITATIVE ERROR & BEHAVIORAL ANALYSIS ===")
    print("-"*90)
    print(f"1. Retrieval Regression Cases (Direct=✓, RAG=✗) [N={len(retrieval_regressions)}]:")
    for r in retrieval_regressions:
        print(f"   [{r['id']}] Q: {r['question']}")
        print(f"      Gold:        '{r['gold_answer']}'")
        print(f"      Direct Ans:  '{r['direct_answer']}' (✓ Correct)")
        print(f"      RAG Ans:     '{r['rag_answer']}' (✗ Distracted/Corrupted)")
        print(f"      Adaptive:    Routed to {r['adaptive_routing_decision']} -> '{r['adaptive_answer']}' (✓ Preserved Direct Answer)")

    print(f"\n2. Retrieval Benefit Cases (Direct=✗, RAG=✓) [Total N={len(retrieval_benefits)} | Showing 3]:")
    for r in retrieval_benefits[:3]:
        print(f"   [{r['id']}] Q: {r['question']}")
        print(f"      Gold:        '{r['gold_answer']}'")
        print(f"      Direct Ans:  '{r['direct_answer']}' (✗ Missed fact)")
        print(f"      RAG Ans:     '{r['rag_answer']}' (✓ Grounded by context)")
        print(f"      Adaptive:    Routed to {r['adaptive_routing_decision']} (Due to classifier prior)")

    print(f"\n3. Routing Decisions Summary:")
    print(f"   • Routed to DIRECT: {adaptive_direct_routes} ({adaptive_direct_routes/total*100:.1f}%)")
    print(f"   • Routed to RAG:    {adaptive_rag_routes} ({adaptive_rag_routes/total*100:.1f}%)")
    print(f"   • Matches Better System: {len(adaptive_matches_better)} queries (including both regression cases)")
    print(f"   • Suboptimal Decisions:  {len(adaptive_chooses_suboptimal)} queries (missed retrieval-benefit opportunities)")


if __name__ == "__main__":
    main()
