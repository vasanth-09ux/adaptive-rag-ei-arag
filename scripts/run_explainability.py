"""
Adaptive-RAG — Explainability & Routing Inspection Application (run_explainability.py)
Makes the routing mechanism, probabilities, retrieval status, and latencies visible and understandable.
Executable independently via: .venv\\Scripts\\python.exe run_explainability.py
"""

import sys
import io
import time
import argparse
import threading
from pathlib import Path
import tkinter as tk
from tkinter import ttk, scrolledtext

# Ensure UTF-8 console output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipelines.adaptive_rag import AdaptiveRAGPipeline


class ExplainabilityApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Adaptive-RAG — Routing Explainability & Decision Inspector")
        self.root.geometry("900x820")
        self.root.minsize(750, 650)
        self.root.configure(bg="#F8FAFC")

        self.pipeline = None
        self.is_evaluating = False

        self._setup_styles()
        self._build_ui()
        self._start_background_init()

    def _setup_styles(self):
        self.style = ttk.Style()
        try:
            self.style.theme_use("clam")
        except Exception:
            pass

        self.style.configure("TFrame", background="#F8FAFC")
        self.style.configure("Card.TFrame", background="#FFFFFF", relief="solid", borderwidth=1)
        self.style.configure("Header.TFrame", background="#0F172A")
        self.style.configure("HeaderTitle.TLabel", font=("Segoe UI", 14, "bold"), foreground="#FFFFFF", background="#0F172A")
        self.style.configure("HeaderSub.TLabel", font=("Segoe UI", 9), foreground="#94A3B8", background="#0F172A")
        self.style.configure("SectionTitle.TLabel", font=("Segoe UI", 11, "bold"), foreground="#1E293B", background="#FFFFFF")
        self.style.configure("MetricLabel.TLabel", font=("Segoe UI", 9), foreground="#64748B", background="#FFFFFF")
        self.style.configure("MetricValue.TLabel", font=("Segoe UI", 11, "bold"), foreground="#0F172A", background="#FFFFFF")
        self.style.configure("DirectBadge.TLabel", font=("Segoe UI", 11, "bold"), foreground="#1D4ED8", background="#DBEAFE", padding=4)
        self.style.configure("RAGBadge.TLabel", font=("Segoe UI", 11, "bold"), foreground="#047857", background="#D1FAE5", padding=4)
        self.style.configure("IdleBadge.TLabel", font=("Segoe UI", 10), foreground="#64748B", background="#E2E8F0", padding=4)
        self.style.configure("Sample.TButton", font=("Segoe UI", 9), padding=4)
        self.style.configure("Inspect.TButton", font=("Segoe UI", 10, "bold"), padding=6)

    def _build_ui(self):
        # 1. Header Bar
        header = ttk.Frame(self.root, style="Header.TFrame", padding=14)
        header.pack(fill=tk.X)

        title = ttk.Label(header, text="Adaptive-RAG Routing & Explainability Inspector", style="HeaderTitle.TLabel")
        title.pack(anchor=tk.W)

        subtitle = ttk.Label(
            header,
            text="Real-time layer-1 contextual routing visualization (Decision boundary, P(Direct) vs P(RAG), Retriever status)",
            style="HeaderSub.TLabel"
        )
        subtitle.pack(anchor=tk.W, pady=(2, 0))

        main_scrollable = ttk.Frame(self.root, padding=12)
        main_scrollable.pack(fill=tk.BOTH, expand=True)

        # 2. Input & Quick Samples Section
        input_card = tk.LabelFrame(main_scrollable, text=" Query Input ", font=("Segoe UI", 10, "bold"), bg="#FFFFFF", fg="#1E293B", padx=10, pady=8)
        input_card.pack(fill=tk.X, pady=(0, 8))

        quick_btns_frame = tk.Frame(input_card, bg="#FFFFFF")
        quick_btns_frame.pack(fill=tk.X, pady=(0, 6))

        tk.Label(quick_btns_frame, text="Quick Test Examples:", font=("Segoe UI", 9, "italic"), bg="#FFFFFF", fg="#64748B").pack(side=tk.LEFT, padx=(0, 6))

        b1 = tk.Button(
            quick_btns_frame,
            text="Parametric (Direct): What is the capital of France?",
            font=("Segoe UI", 8),
            bg="#EFF6FF",
            fg="#1D4ED8",
            relief=tk.GROOVE,
            command=lambda: self._set_query("What is the capital of France?")
        )
        b1.pack(side=tk.LEFT, padx=3)

        b2 = tk.Button(
            quick_btns_frame,
            text="Domain (RAG): In what year was the NVIDIA GeForce GTX 1650 launched?",
            font=("Segoe UI", 8),
            bg="#ECFDF5",
            fg="#047857",
            relief=tk.GROOVE,
            command=lambda: self._set_query("In what year was the NVIDIA GeForce GTX 1650 launched?")
        )
        b2.pack(side=tk.LEFT, padx=3)

        input_row = tk.Frame(input_card, bg="#FFFFFF")
        input_row.pack(fill=tk.X)

        self.query_entry = tk.Entry(input_row, font=("Segoe UI", 10), bg="#F8FAFC", relief=tk.SOLID, bd=1)
        self.query_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8), ipady=5)
        self.query_entry.bind("<Return>", lambda e: self._on_inspect())

        self.inspect_btn = tk.Button(
            input_row,
            text="Inspect Routing",
            font=("Segoe UI", 10, "bold"),
            bg="#2563EB",
            fg="#FFFFFF",
            relief=tk.FLAT,
            padx=12,
            pady=4,
            command=self._on_inspect
        )
        self.inspect_btn.pack(side=tk.RIGHT)

        # 3. Explainability Metrics Grid (Decision, Probabilities, Retrieval, Latencies)
        metrics_card = tk.LabelFrame(main_scrollable, text=" Routing & Probability Explainability ", font=("Segoe UI", 10, "bold"), bg="#FFFFFF", fg="#1E293B", padx=10, pady=8)
        metrics_card.pack(fill=tk.X, pady=(0, 8))

        # Row 1: Decision, Threshold, P(Direct), P(RAG)
        row1 = tk.Frame(metrics_card, bg="#FFFFFF")
        row1.pack(fill=tk.X, pady=4)

        # Decision Badge
        d_box = tk.Frame(row1, bg="#F1F5F9", padx=10, pady=6, relief=tk.RIDGE, bd=1)
        d_box.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        tk.Label(d_box, text="Routing Decision", font=("Segoe UI", 8), bg="#F1F5F9", fg="#64748B").pack(anchor=tk.W)
        self.decision_label = tk.Label(d_box, text="AWAITING QUERY", font=("Segoe UI", 12, "bold"), bg="#F1F5F9", fg="#475569")
        self.decision_label.pack(anchor=tk.W)

        # Threshold Badge
        t_box = tk.Frame(row1, bg="#F1F5F9", padx=10, pady=6, relief=tk.RIDGE, bd=1)
        t_box.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        tk.Label(t_box, text="Decision Threshold", font=("Segoe UI", 8), bg="#F1F5F9", fg="#64748B").pack(anchor=tk.W)
        self.threshold_label = tk.Label(t_box, text="50.0% (0.5000)", font=("Segoe UI", 12, "bold"), bg="#F1F5F9", fg="#0F172A")
        self.threshold_label.pack(anchor=tk.W)

        # P(Direct)
        pdir_box = tk.Frame(row1, bg="#EFF6FF", padx=10, pady=6, relief=tk.RIDGE, bd=1)
        pdir_box.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        tk.Label(pdir_box, text="P(Direct / Parametric)", font=("Segoe UI", 8), bg="#EFF6FF", fg="#1E40AF").pack(anchor=tk.W)
        self.p_direct_label = tk.Label(pdir_box, text="-- %", font=("Segoe UI", 12, "bold"), bg="#EFF6FF", fg="#1D4ED8")
        self.p_direct_label.pack(anchor=tk.W)

        # P(RAG)
        prag_box = tk.Frame(row1, bg="#ECFDF5", padx=10, pady=6, relief=tk.RIDGE, bd=1)
        prag_box.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        tk.Label(prag_box, text="P(RAG / External)", font=("Segoe UI", 8), bg="#ECFDF5", fg="#065F46").pack(anchor=tk.W)
        self.p_rag_label = tk.Label(prag_box, text="-- %", font=("Segoe UI", 12, "bold"), bg="#ECFDF5", fg="#047857")
        self.p_rag_label.pack(anchor=tk.W)

        # Visual Probability Bar Gauge
        gauge_frame = tk.Frame(metrics_card, bg="#FFFFFF")
        gauge_frame.pack(fill=tk.X, pady=(6, 4))
        tk.Label(gauge_frame, text="Probability Distribution vs Decision Threshold (50%):", font=("Segoe UI", 8, "italic"), bg="#FFFFFF", fg="#64748B").pack(anchor=tk.W)

        self.gauge_canvas = tk.Canvas(gauge_frame, height=26, bg="#E2E8F0", highlightthickness=0)
        self.gauge_canvas.pack(fill=tk.X, pady=2)
        self._draw_empty_gauge()

        # Row 2: Retrieval Inspector
        ret_card = tk.LabelFrame(main_scrollable, text=" Retrieval & Corpus Inspection ", font=("Segoe UI", 10, "bold"), bg="#FFFFFF", fg="#1E293B", padx=10, pady=8)
        ret_card.pack(fill=tk.X, pady=(0, 8))

        ret_row = tk.Frame(ret_card, bg="#FFFFFF")
        ret_row.pack(fill=tk.X)

        # Retrieval Status
        rs_box = tk.Frame(ret_row, bg="#F8FAFC", padx=8, pady=4, relief=tk.RIDGE, bd=1)
        rs_box.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=3)
        tk.Label(rs_box, text="Retrieval Status", font=("Segoe UI", 8), bg="#F8FAFC", fg="#64748B").pack(anchor=tk.W)
        self.ret_status_label = tk.Label(rs_box, text="Not Triggered", font=("Segoe UI", 10, "bold"), bg="#F8FAFC", fg="#334155")
        self.ret_status_label.pack(anchor=tk.W)

        # Top Document
        doc_box = tk.Frame(ret_row, bg="#F8FAFC", padx=8, pady=4, relief=tk.RIDGE, bd=1)
        doc_box.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=3)
        tk.Label(doc_box, text="Top Retrieved Document", font=("Segoe UI", 8), bg="#F8FAFC", fg="#64748B").pack(anchor=tk.W)
        self.top_doc_label = tk.Label(doc_box, text="N/A", font=("Segoe UI", 10, "bold"), bg="#F8FAFC", fg="#334155")
        self.top_doc_label.pack(anchor=tk.W)

        # Top Similarity Score
        sim_box = tk.Frame(ret_row, bg="#F8FAFC", padx=8, pady=4, relief=tk.RIDGE, bd=1)
        sim_box.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=3)
        tk.Label(sim_box, text="Cosine Similarity Score", font=("Segoe UI", 8), bg="#F8FAFC", fg="#64748B").pack(anchor=tk.W)
        self.sim_score_label = tk.Label(sim_box, text="N/A", font=("Segoe UI", 10, "bold"), bg="#F8FAFC", fg="#334155")
        self.sim_score_label.pack(anchor=tk.W)

        # Row 3: Latency Breakdown
        lat_card = tk.LabelFrame(main_scrollable, text=" Latency Breakdown ", font=("Segoe UI", 10, "bold"), bg="#FFFFFF", fg="#1E293B", padx=10, pady=8)
        lat_card.pack(fill=tk.X, pady=(0, 8))

        lat_row = tk.Frame(lat_card, bg="#FFFFFF")
        lat_row.pack(fill=tk.X)

        l_ret = tk.Frame(lat_row, bg="#F8FAFC", padx=8, pady=4, relief=tk.RIDGE, bd=1)
        l_ret.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=3)
        tk.Label(l_ret, text="Retrieval Latency", font=("Segoe UI", 8), bg="#F8FAFC", fg="#64748B").pack(anchor=tk.W)
        self.ret_lat_label = tk.Label(l_ret, text="0.00 ms", font=("Segoe UI", 10, "bold"), bg="#F8FAFC", fg="#0F172A")
        self.ret_lat_label.pack(anchor=tk.W)

        l_gen = tk.Frame(lat_row, bg="#F8FAFC", padx=8, pady=4, relief=tk.RIDGE, bd=1)
        l_gen.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=3)
        tk.Label(l_gen, text="Generation Latency", font=("Segoe UI", 8), bg="#F8FAFC", fg="#64748B").pack(anchor=tk.W)
        self.gen_lat_label = tk.Label(l_gen, text="0.00 ms", font=("Segoe UI", 10, "bold"), bg="#F8FAFC", fg="#0F172A")
        self.gen_lat_label.pack(anchor=tk.W)

        l_tot = tk.Frame(lat_row, bg="#F8FAFC", padx=8, pady=4, relief=tk.RIDGE, bd=1)
        l_tot.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=3)
        tk.Label(l_tot, text="Total End-to-End Latency", font=("Segoe UI", 8), bg="#F8FAFC", fg="#64748B").pack(anchor=tk.W)
        self.tot_lat_label = tk.Label(l_tot, text="0.00 ms", font=("Segoe UI", 10, "bold"), bg="#F8FAFC", fg="#2563EB")
        self.tot_lat_label.pack(anchor=tk.W)

        # 4. Final Answer Output Box
        ans_card = tk.LabelFrame(main_scrollable, text=" Final Generated Answer ", font=("Segoe UI", 10, "bold"), bg="#FFFFFF", fg="#1E293B", padx=10, pady=8)
        ans_card.pack(fill=tk.BOTH, expand=True)

        self.answer_text = scrolledtext.ScrolledText(
            ans_card,
            wrap=tk.WORD,
            font=("Segoe UI", 10),
            bg="#F8FAFC",
            fg="#0F172A",
            relief=tk.FLAT,
            height=6,
            state=tk.DISABLED
        )
        self.answer_text.pack(fill=tk.BOTH, expand=True)

        # Status Footer
        self.status_label = tk.Label(
            self.root,
            text="Initializing models onto GTX 1650...",
            font=("Segoe UI", 8),
            bg="#F8FAFC",
            fg="#64748B",
            anchor=tk.W,
            padx=12,
            pady=4
        )
        self.status_label.pack(fill=tk.X, side=tk.BOTTOM)

    def _draw_empty_gauge(self):
        self.gauge_canvas.delete("all")
        width = self.gauge_canvas.winfo_width()
        if width <= 1:
            width = 850
        h = 26
        # Background bar
        self.gauge_canvas.create_rectangle(0, 0, width, h, fill="#E2E8F0", outline="")
        # Threshold marker at 50%
        mid = width * 0.5
        self.gauge_canvas.create_line(mid, 0, mid, h, fill="#0F172A", width=2, dash=(3, 2))
        self.gauge_canvas.create_text(mid, h / 2, text="Threshold (50%)", fill="#475569", font=("Segoe UI", 7, "bold"))

    def _update_gauge(self, p_direct: float, p_rag: float):
        self.gauge_canvas.delete("all")
        self.gauge_canvas.update_idletasks()
        width = self.gauge_canvas.winfo_width()
        if width <= 1:
            width = 850
        h = 26

        # Draw Direct portion (left, blue)
        direct_w = width * p_direct
        self.gauge_canvas.create_rectangle(0, 0, direct_w, h, fill="#3B82F6", outline="")
        if direct_w > 60:
            self.gauge_canvas.create_text(direct_w / 2, h / 2, text=f"Direct: {p_direct*100:.1f}%", fill="#FFFFFF", font=("Segoe UI", 8, "bold"))

        # Draw RAG portion (right, green)
        rag_w = width - direct_w
        self.gauge_canvas.create_rectangle(direct_w, 0, width, h, fill="#10B981", outline="")
        if rag_w > 60:
            self.gauge_canvas.create_text(direct_w + (rag_w / 2), h / 2, text=f"RAG: {p_rag*100:.1f}%", fill="#FFFFFF", font=("Segoe UI", 8, "bold"))

        # Draw threshold marker at 50%
        mid = width * 0.5
        self.gauge_canvas.create_line(mid, 0, mid, h, fill="#DC2626", width=2)
        self.gauge_canvas.create_text(mid, h - 6, text="50%", fill="#DC2626", font=("Segoe UI", 7, "bold"))

    def _set_query(self, text: str):
        self.query_entry.delete(0, tk.END)
        self.query_entry.insert(0, text)

    def _start_background_init(self):
        def _loader():
            try:
                self.pipeline = AdaptiveRAGPipeline(mode="adaptive")
                self.root.after(0, self._on_init_success)
            except Exception as e:
                self.root.after(0, lambda: self._on_init_error(str(e)))

        t = threading.Thread(target=_loader, daemon=True)
        t.start()

    def _on_init_success(self):
        self.status_label.config(text="Models loaded and ready on GTX 1650 (classifier_expanded.pt active)")

    def _on_init_error(self, err: str):
        self.status_label.config(text=f"Initialization Error: {err}", fg="#DC2626")

    def _on_inspect(self):
        if self.is_evaluating or not self.pipeline:
            return
        query = self.query_entry.get().strip()
        if not query:
            return

        self.is_evaluating = True
        self.inspect_btn.config(state=tk.DISABLED)
        self.status_label.config(text=f"Evaluating routing decision for: '{query}'...")

        def _worker():
            try:
                result = self.pipeline.answer(query)
                self.root.after(0, lambda: self._display_result(result))
            except Exception as e:
                self.root.after(0, lambda: self._display_error(str(e)))

        threading.Thread(target=_worker, daemon=True).start()

    def _display_result(self, res: dict):
        self.is_evaluating = False
        self.inspect_btn.config(state=tk.NORMAL)
        self.status_label.config(text="Inference completed successfully.")

        decision = res.get("routing_decision", "UNKNOWN")
        p_rag = res.get("classifier_probability", 0.0)
        p_direct = round(1.0 - p_rag, 4)

        # Update Decision Badge
        if decision == "RAG":
            self.decision_label.config(text="ROUTED TO RAG", fg="#047857", bg="#D1FAE5")
        else:
            self.decision_label.config(text="ROUTED TO DIRECT", fg="#1D4ED8", bg="#DBEAFE")

        # Update Probabilities
        self.p_direct_label.config(text=f"{p_direct * 100:.2f}% ({p_direct:.4f})")
        self.p_rag_label.config(text=f"{p_rag * 100:.2f}% ({p_rag:.4f})")
        self._update_gauge(p_direct, p_rag)

        # Update Retrieval Details
        docs = res.get("retrieved_documents", [])
        if decision == "RAG" and docs:
            self.ret_status_label.config(text="TRIGGERED (Retriever Active)", fg="#047857")
            top_doc = docs[0].get("doc_name", "Unknown")
            top_score = docs[0].get("score", 0.0)
            self.top_doc_label.config(text=top_doc, fg="#0F172A")
            self.sim_score_label.config(text=f"{top_score:.4f} (Cosine)", fg="#047857")
        else:
            self.ret_status_label.config(text="BYPASSED (Parametric Memory)", fg="#1D4ED8")
            self.top_doc_label.config(text="N/A (Skipped)", fg="#64748B")
            self.sim_score_label.config(text="N/A", fg="#64748B")

        # Update Latency Breakdown
        self.ret_lat_label.config(text=f"{res.get('retrieval_latency_ms', 0.0):.2f} ms")
        self.gen_lat_label.config(text=f"{res.get('generation_latency_ms', 0.0):.2f} ms")
        self.tot_lat_label.config(text=f"{res.get('total_latency_ms', 0.0):.2f} ms")

        # Update Answer Text
        self.answer_text.config(state=tk.NORMAL)
        self.answer_text.delete("1.0", tk.END)
        self.answer_text.insert(tk.END, res.get("answer", ""))
        self.answer_text.config(state=tk.DISABLED)

    def _display_error(self, err: str):
        self.is_evaluating = False
        self.inspect_btn.config(state=tk.NORMAL)
        self.status_label.config(text=f"Error: {err}", fg="#DC2626")


def run_explainability_test():
    """Automated verification test for the explainability pipeline."""
    print("=" * 60)
    print("ADAPTIVE-RAG: EXPLAINABILITY PIPELINE TEST")
    print("=" * 60)

    print("\n1. Initializing Adaptive-RAG Pipeline...")
    pipeline = AdaptiveRAGPipeline(mode="adaptive")
    print("   Pipeline successfully loaded with classifier_expanded.pt")

    # Test Query 1: Parametric / Direct
    q1 = "What is the capital of France?"
    print(f"\n2. Testing Query 1: '{q1}'")
    res1 = pipeline.answer(q1)
    p_rag1 = res1["classifier_probability"]
    p_dir1 = round(1.0 - p_rag1, 4)
    decision1 = res1["routing_decision"]
    print(f"   P(Direct): {p_dir1*100:.2f}% | P(RAG): {p_rag1*100:.2f}% | Threshold: 50.00%")
    print(f"   Routing Decision: {decision1}")
    print(f"   Retrieval Status: {'TRIGGERED' if decision1 == 'RAG' else 'BYPASSED'}")
    print(f"   Retrieval Latency: {res1['retrieval_latency_ms']:.2f} ms")
    print(f"   Generation Latency: {res1['generation_latency_ms']:.2f} ms")
    print(f"   Total Latency: {res1['total_latency_ms']:.2f} ms")
    print(f"   Answer: {res1['answer'].strip()[:100]}...")
    assert decision1 == "DIRECT", f"Expected DIRECT for q1, got {decision1}"
    print("   -> [PASS] Query 1 correctly routed to DIRECT")

    # Test Query 2: Domain / RAG
    q2 = "In what year was the NVIDIA GeForce GTX 1650 launched?"
    print(f"\n3. Testing Query 2: '{q2}'")
    res2 = pipeline.answer(q2)
    p_rag2 = res2["classifier_probability"]
    p_dir2 = round(1.0 - p_rag2, 4)
    decision2 = res2["routing_decision"]
    docs2 = res2.get("retrieved_documents", [])
    top_doc2 = docs2[0]["doc_name"] if docs2 else "None"
    top_score2 = docs2[0]["score"] if docs2 else 0.0
    print(f"   P(Direct): {p_dir2*100:.2f}% | P(RAG): {p_rag2*100:.2f}% | Threshold: 50.00%")
    print(f"   Routing Decision: {decision2}")
    print(f"   Retrieval Status: {'TRIGGERED' if decision2 == 'RAG' else 'BYPASSED'}")
    print(f"   Top Retrieved Document: {top_doc2} (Score: {top_score2:.4f})")
    print(f"   Retrieval Latency: {res2['retrieval_latency_ms']:.2f} ms")
    print(f"   Generation Latency: {res2['generation_latency_ms']:.2f} ms")
    print(f"   Total Latency: {res2['total_latency_ms']:.2f} ms")
    print(f"   Answer: {res2['answer'].strip()[:100]}...")
    assert decision2 == "RAG", f"Expected RAG for q2, got {decision2}"
    print("   -> [PASS] Query 2 correctly routed to RAG")

    print("\n" + "=" * 60)
    print("ALL EXPLAINABILITY CHECKS PASSED SUCCESSFULLY!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Adaptive-RAG Explainability & Routing Inspector")
    parser.add_argument("--test", action="store_true", help="Run automated routing verification on test queries")
    parser.add_argument("--query", type=str, default=None, help="Inspect a single query directly from CLI")
    args = parser.parse_args()

    if args.test:
        run_explainability_test()
        return

    if args.query:
        pipeline = AdaptiveRAGPipeline(mode="adaptive")
        res = pipeline.answer(args.query)
        p_rag = res["classifier_probability"]
        p_dir = round(1.0 - p_rag, 4)
        print(f"Query: {args.query}")
        print(f"Decision: {res['routing_decision']} (P(Direct)={p_dir*100:.2f}%, P(RAG)={p_rag*100:.2f}%)")
        if res["routing_decision"] == "RAG" and res["retrieved_documents"]:
            print(f"Top Doc: {res['retrieved_documents'][0]['doc_name']} (Similarity: {res['retrieved_documents'][0]['score']:.4f})")
        print(f"Latency: Retrieval={res['retrieval_latency_ms']:.1f}ms, Gen={res['generation_latency_ms']:.1f}ms, Total={res['total_latency_ms']:.1f}ms")
        print(f"Answer:\n{res['answer']}")
        return

    root = tk.Tk()
    app = ExplainabilityApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
