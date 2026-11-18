# Reproduction Guide: Adaptive-RAG (EI-ARAG)

This guide provides step-by-step instructions for executing and reproducing the evaluations, interactive GUI tools, and classification models in this repository.

---

## 1. System Requirements

### Hardware Profile
- **GPU (Recommended for live LLM inference):** NVIDIA GPU with $\ge 4\text{ GB}$ dedicated VRAM (e.g., GTX 1650, RTX 3050, or better).
- **CPU:** Multi-core x86_64 host CPU (Intel Core i5/i7 or AMD Ryzen).
- **RAM:** Minimum 16 GB system memory.
- **Disk:** ~10 GB free space (to accommodate Qwen2.5-1.5B model weights cached from Hugging Face).

*(Note: If a CUDA-enabled GPU is unavailable, offline dashboard reproduction and classifier evaluation run 100% on CPU without requiring LLM inference).*

---

## 2. Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/vasanth-09ux/adaptive-rag-ei-arag.git
   cd adaptive-rag-ei-arag
   ```

2. **Create and Activate a Virtual Environment:**
   ```bash
   python -m venv .venv

   # On Windows:
   .venv\Scripts\activate

   # On Linux/macOS:
   source .venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 3. Interactive Graphical User Interfaces (GUIs)

The repository provides three lightweight desktop applications built with Python Tkinter:

### 3.1 Normal Chat Application (`run_chat.py`)
An interactive desktop chatbot with real-time routing telemetry. For every query entered, the application displays whether retrieval was triggered, the classifier confidence, and the response.
```bash
python scripts/run_chat.py
```

### 3.2 Explainability & Routing Inspector (`run_explainability.py`)
Provides granular visibility into routing decisions. Displays the extracted Layer-1 vector norm, softmax probabilities ($P(\text{Direct})$ vs. $P(\text{RAG})$), retrieval latency breakdown, and preview of top-3 retrieved passages.
```bash
python scripts/run_explainability.py
```

### 3.3 Routing & Evaluation Dashboard (`run_routing_dashboard.py`)
Visualizes the frozen benchmark evaluation results, comparative accuracy bars, latency reductions, and confusion matrices directly from the saved evaluation artifacts without requiring LLM execution.
```bash
python scripts/run_routing_dashboard.py
```

---

## 4. End-to-End Evaluation & Benchmarks

### 4.1 Quick Validation Smoke Test (15 Questions)
Executes a fast validation smoke test on the 15-question validation split:
```bash
python scripts/run_validation_smoke_test.py
```

### 4.2 Full 100-Query Benchmark Evaluation
Executes the comprehensive evaluation across Direct LLM, Adaptive-RAG, and Always-RAG modes, calculating accuracy metrics and latencies:
```bash
python scripts/run_full_evaluation.py
```

---

## 5. Model Training & Embedding Extraction

### 5.1 Training the Expanded Routing Classifier
Trains the 49,250-parameter MLP classifier on the expanded pool of 310 samples (using pre-extracted Layer-1 representations):
```bash
python scripts/train_expanded_classifier.py
```
*Expected Output:* Trains for 30 epochs with AdamW, outputs loss progression, and logs test set accuracy (71.11%), Label-1 precision (66.67%), and recall (13.33%).

### 5.2 Extracting Layer-1 Representations
To re-extract Layer-1 hidden states directly from Qwen2.5-1.5B for a custom dataset:
```bash
python scripts/extract_expanded_embeddings.py
```
*(Requires a CUDA GPU with $\ge 4\text{ GB}$ VRAM).*
