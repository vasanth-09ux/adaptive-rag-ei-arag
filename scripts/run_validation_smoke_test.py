import sys
import io
import json
from pathlib import Path
# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from src.classifier.data_generator import DataGenerator

def main():
    print("=== Running EI-ARAG Validation Smoke Test (15 Questions) ===")
    
    benchmark_path = PROJECT_ROOT / "data" / "benchmark_queries.json"
    with open(benchmark_path, "r", encoding="utf-8") as f:
        all_queries = json.load(f)

    # Filter ONLY validation split
    val_queries = [q for q in all_queries if q.get("split") == "validation"]
    print(f"Loaded {len(val_queries)} validation queries (out of {len(all_queries)} total).")
    assert len(val_queries) == 15, f"Expected 15 validation queries, got {len(val_queries)}"

    output_path = PROJECT_ROOT / "data" / "validation_smoke_test_results.json"
    
    generator = DataGenerator()
    stats = generator.generate_labeled_dataset(
        qa_pairs=val_queries,
        output_file=output_path
    )

    records = stats["records"]
    total = len(records)
    label_0_count = stats["label_0_count"]
    label_1_count = stats["label_1_count"]
    direct_acc = stats["direct_accuracy"]
    rag_acc = stats["rag_accuracy"]

    # Latency calculations
    direct_latencies = [r["latency_ms"]["direct_generation"] for r in records]
    rag_latencies = [r["latency_ms"]["rag_total"] for r in records]
    avg_direct_lat = sum(direct_latencies) / total if total > 0 else 0
    avg_rag_lat = sum(rag_latencies) / total if total > 0 else 0

    # Categorize cases
    beneficial_cases = [r for r in records if r["label"] == 1]
    regression_cases = [r for r in records if r["direct_correct"] and not r["rag_correct"]]
    both_correct = [r for r in records if r["direct_correct"] and r["rag_correct"]]
    both_failed = [r for r in records if not r["direct_correct"] and not r["rag_correct"]]

    print("\n" + "="*60)
    print("=== VALIDATION SMOKE-TEST SUMMARY (15 QUESTIONS) ===")
    print("="*60)
    print(f"1. Total Questions Evaluated:  {total}")
    print(f"2. Direct Model Accuracy:      {direct_acc * 100:.2f}% ({sum(1 for r in records if r['direct_correct'])}/{total})")
    print(f"3. RAG Model Accuracy:         {rag_acc * 100:.2f}% ({sum(1 for r in records if r['rag_correct'])}/{total})")
    print(f"4. Label 0 Count:              {label_0_count} ({label_0_count/total*100:.1f}%)")
    print(f"5. Label 1 Count:              {label_1_count} ({label_1_count/total*100:.1f}%)")
    print(f"6. Both Correct (Label 0):     {len(both_correct)}")
    print(f"7. Both Failed (Label 0):      {len(both_failed)}")
    print(f"8. Retrieval Regressions:      {len(regression_cases)}")
    print(f"9. Avg Direct Latency:         {avg_direct_lat:.2f} ms")
    print(f"10. Avg RAG Total Latency:     {avg_rag_lat:.2f} ms")

    print("\n" + "="*60)
    print("=== RETRIEVAL-BENEFICIAL CASES (Direct=False -> RAG=True | Label=1) ===")
    print("="*60)
    if not beneficial_cases:
        print("None.")
    else:
        for r in beneficial_cases:
            print(f"\nID: {r['id']} | Category: {r['category']} | Source: {r.get('source')}")
            print(f"Question:      {r['question']}")
            print(f"Gold Answer:   {r['gold_answer']}")
            print(f"Direct Answer: {r['direct_answer']}")
            print(f"RAG Answer:    {r['rag_answer']}")
            print(f"Retrieved:     {[s['doc_name'] + ' (score: ' + str(s['score']) + ')' for s in r['retrieved_sources']]}")

    print("\n" + "="*60)
    print("=== RETRIEVAL-REGRESSION CASES (Direct=True -> RAG=False | Label=0) ===")
    print("="*60)
    if not regression_cases:
        print("None.")
    else:
        for r in regression_cases:
            print(f"\nID: {r['id']} | Category: {r['category']} | Source: {r.get('source')}")
            print(f"Question:      {r['question']}")
            print(f"Gold Answer:   {r['gold_answer']}")
            print(f"Direct Answer: {r['direct_answer']}")
            print(f"RAG Answer:    {r['rag_answer']}")

    print("\n" + "="*60)
    print("=== AUDIT OF ALL 15 EVALUATED QUESTIONS ===")
    print("="*60)
    for r in records:
        status_str = "LABEL 1 (RAG Helped)" if r["label"] == 1 else ("REGRESSION" if (r["direct_correct"] and not r["rag_correct"]) else "LABEL 0")
        print(f"[{r['id']}] [{r['category']:<10}] Direct:{str(r['direct_correct']):<5} | RAG:{str(r['rag_correct']):<5} | Label:{r['label']} | {status_str}")
        print(f"     Q: {r['question']}")
        print(f"     Gold: '{r['gold_answer']}'")
        print(f"     Direct Ans: '{r['direct_answer']}'")
        print(f"     RAG Ans:    '{r['rag_answer']}'")
        print()

if __name__ == "__main__":
    main()
