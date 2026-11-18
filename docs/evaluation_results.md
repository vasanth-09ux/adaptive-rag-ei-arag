# Frozen Experimental Evaluation Results: EI-ARAG

All quantitative results documented herein are extracted directly from the verified evaluation artifacts:
- `data/expanded_evaluation_summary.json`
- `data/evaluation_summary.json`
- `models/classifier_expanded_metadata.json`
- `data/evaluation_results_100.json`

---

## 1. Primary Benchmark Performance (100 Queries)

The primary evaluation benchmark consists of 100 stratified queries spanning three distinct difficulty profiles:
- **Parametric (35 queries):** Core world knowledge accessible within the LLM's pre-trained weights.
- **Retrieval-Dependent (45 queries):** Domain-specific, tail factual knowledge requiring external document verification.
- **Distractors (20 queries):** Questions designed to test robustness against ambiguous or deceptive phrasing.

### 1.1 Comparative System Summary Across Independent Baselines

| Evaluation Dimension | Direct LLM (Parametric) | Adaptive-RAG (EI-ARAG) | Always-RAG (Baseline) | Comparison: Adaptive vs. Direct LLM | Comparison: Adaptive vs. Always-RAG |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Accuracy (%)** | **59.00%** | **64.00%** | **90.00%** | **+5.00 pp** (Accuracy Gain) | **-26.00 pp** (Accuracy Trade-Off) |
| **Correct Answers** | 59 / 100 | 64 / 100 | 90 / 100 | +5 correct answers | -26 correct answers |
| **Incorrect Answers** | 41 / 100 | 36 / 100 | 10 / 100 | -5 errors | +26 errors |
| **Average Latency (ms)** | **1,466.09 ms** | **2,070.16 ms** | **8,665.74 ms** | +604.07 ms (+41.2% routing overhead) | **-6,595.58 ms (76.11% latency reduction)** |
| **Median Latency (ms)** | 1,082.07 ms | 1,320.42 ms | 8,414.09 ms | +238.35 ms | -7,093.67 ms |
| **Minimum Latency (ms)** | 653.49 ms | 653.49 ms | 6,540.21 ms | 0.00 ms | -5,886.72 ms |
| **Maximum Latency (ms)** | 4,218.30 ms | 15,469.29 ms | 18,920.10 ms | +11,250.99 ms | -3,450.81 ms |
| **Retrieval Calls** | 0 / 100 (0.0%) | 7 / 100 (7.0%) | 100 / 100 (100.0%) | +7 retrieval calls | **-93 retrieval calls (93.0% calls saved)** |
| **Direct-Path Queries** | 100 / 100 (100.0%) | 93 / 100 (93.0%) | 0 / 100 (0.0%) | -7 direct queries | +93 direct queries |

---

### 1.2 Mathematical Derivations of Latency Changes

1. **Adaptive-RAG vs. Always-RAG Latency Savings:**
   - Absolute reduction:
     $$\Delta t = 8,665.74\text{ ms} - 2,070.16\text{ ms} = \mathbf{6,595.58\text{ ms saved per query}}$$
   - Relative reduction:
     $$\text{Latency Reduction} = \frac{8,665.74 - 2,070.16}{8,665.74} \times 100\% = \frac{6,595.58}{8,665.74} \times 100\% = \mathbf{76.11098\% \approx 76.11\%}$$
   - Latency multiplier (Always-RAG vs. Direct LLM):
     $$\frac{8,665.74\text{ ms}}{1,466.09\text{ ms}} = \mathbf{5.91\times \text{ slower than direct generation}}$$

2. **Adaptive-RAG vs. Direct LLM Overhead:**
   - Absolute overhead:
     $$\Delta t_{\text{direct}} = 2,070.16\text{ ms} - 1,466.09\text{ ms} = \mathbf{+604.07\text{ ms}}$$
   - Relative increase:
     $$\frac{604.07}{1,466.09} \times 100\% = \mathbf{+41.20\%}$$
     *(This overhead is composed of Layer-1 extraction and mean pooling on GPU + CPU MLP inference across all 100 queries, plus full context-expanded retrieval and generation for the 7 routed queries).*

---

## 2. Granular Outcome Breakdown of the Seven RAG-Routed Queries

Across the 100-query benchmark, the expanded routing classifier ($P(\text{RAG}) \ge 0.5000$) triggered retrieval for exactly **7 queries**.

### 2.1 Complete Ledger of Routed Queries

| Query ID | Question Prompt | Category | Classifier $P(\text{RAG})$ | Direct LLM Output | RAG-Augmented Output | System Outcome |
| :---: | :--- | :---: | :---: | :--- | :--- | :---: |
| **Q006** | In which year was the NVIDIA GeForce GTX 1650 launched? | Retrieval | **0.5833** | ❌ Guess: "2018" | ✅ Grounded: "2019" | **IMPROVEMENT (+1)** |
| **Q009** | How many CUDA cores are contained in the GeForce GTX 1650? | Distractor | **0.5671** | ❌ Guess: "2304" | ✅ Grounded: "896" | **IMPROVEMENT (+1)** |
| **Q031** | What was the exact date of Alan Turing's birth? | Retrieval | **0.5492** | ✅ "June 23, 1912" | ✅ "June 23, 1912" | **NEUTRAL (0)** |
| **Q035** | In which year was Alan Turing elected a Fellow of the Royal Society? | Retrieval | **0.5691** | ❌ Guess: "1950" | ✅ Grounded: "1951" | **IMPROVEMENT (+1)** |
| **Q045** | How many minutes after Neil Armstrong did Buzz Aldrin step onto lunar surface? | Retrieval | **0.5708** | ❌ Guess: "14 minutes" | ✅ Grounded: "19 minutes" | **IMPROVEMENT (+1)** |
| **Q086** | What is the approximate total length of the Nile River in kilometers? | Retrieval | **0.5608** | ✅ "6,650 km" | ✅ "6,650 km" | **NEUTRAL (0)** |
| **Q097** | Which form of ER is studded with ribosomes for protein synthesis? | Distractor | **0.5457** | ❌ Guess: "Smooth ER" | ✅ Grounded: "Rough ER" | **IMPROVEMENT (+1)** |

### 2.2 System Outcome Classification
The outcome across the 7 routed queries is accurately described as:
- **5 Improvements (71.43% of routed queries):** Direct generation produced an incorrect answer, and RAG retrieval corrected it to the ground truth.
- **2 Neutral Outcomes (28.57% of routed queries):** Direct generation was already correct; retrieval was triggered unnecessarily, incurring latency overhead but causing zero accuracy harm.
- **0 Regressions (0.00% of routed queries):** There were zero instances where retrieval caused the model to degrade an otherwise correct answer.

*(Note on Nomenclature: This 5/7 outcome distribution represents an end-to-end system utility measurement, not standard machine learning classifier precision. Conventional ML precision is reported separately below on the held-out test split).*

---

## 3. Classifier Performance: Original vs. Expanded

Standard machine learning evaluation metrics measured on the independent splits:

| Dimension / Metric | Original Classifier (`classifier_small.pt`) | Expanded Classifier (`classifier_expanded.pt`) | Engineering Significance |
| :--- | :---: | :---: | :--- |
| **Total Labeled Samples** | 100 samples | 310 samples | $+210\%$ dataset expansion |
| **Training Split ($N_{\text{train}}$)** | 70 samples | 220 samples | $+214\%$ increase in training signal |
| **Validation Split ($N_{\text{val}}$)** | 15 samples | 45 samples | 3-fold larger validation set |
| **Test Split ($N_{\text{test}}$)** | 15 samples | 45 samples | 3-fold larger held-out test set |
| **Label-1 Training Samples** | 20 samples | 70 samples | Tripled positive training representations |
| **Overall Test Accuracy** | 73.33% ($11 / 15$) | **71.11%** ($32 / 45$) | Robust generalization over larger test pool |
| **Test Label-1 Precision** | 0.00% ($0 / 0$) | **66.67%** ($2 / 3$) | Resolved zero-precision collapse |
| **Test Label-1 Recall** | 0.00% ($0 / 4$) | **13.33%** ($2 / 15$) | Overcame majority-class collapse |
| **Test Label-1 F1-Score** | 0.0000 | **0.2222** | Stable minority class convergence |

---

## 4. Visual Artifact Reference

The repository preserves all high-resolution figures in `outputs/figures/`:
- `Fig3.1.png`: Detailed EI-ARAG system architecture.
- `figure_3_3_loss_curves.png`: Training & validation loss convergence.
- `figure_4_1_benchmark_categories.png`: Categorical distribution across benchmark queries.
- `figure_5_1_cm_original.png` & `figure_5_2_cm_expanded.png`: Confusion matrices comparing original vs. expanded classifier.
- `figure_5_3_cm_side_by_side.png`: Side-by-side confusion matrix comparison.
- `figure_5_5_overall_accuracy.png`: End-to-end accuracy comparison bar chart.
- `figure_5_6_mean_latency.png`: Mean latency comparison across pipelines.
- `figure_5_7_latency_decomposition.png`: Microscopic breakdown of latency components.
- `figure_5_8_category_accuracy.png`: Category-wise accuracy breakdown.
- `figure_5_9_routed_query_analysis.png`: Probability distributions and outcomes for routed queries.
- `figure_6_1_accuracy_latency_comparison.png`: Dual-axis accuracy vs. latency trade-off frontier.
