"""
Adaptive-RAG — Normal Chat Application (run_chat.py)
A clean, interactive chatbot interface for the Adaptive-RAG pipeline.
Executable independently via: .venv\\Scripts\\python.exe run_chat.py
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

class AdaptiveChatApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Adaptive-RAG — Normal Chat")
        self.root.geometry("750x720")
        self.root.minsize(600, 500)
        self.root.configure(bg="#F1F5F9")

        self.pipeline = None
        self.is_generating = False

        self._setup_styles()
        self._build_ui()
        self._start_background_init()

    def _setup_styles(self):
        self.style = ttk.Style()
        try:
            self.style.theme_use("clam")
        except Exception:
            pass

        self.style.configure("TFrame", background="#F1F5F9")
        self.style.configure("Header.TFrame", background="#1E293B")
        self.style.configure("HeaderTitle.TLabel", font=("Segoe UI", 14, "bold"), foreground="#FFFFFF", background="#1E293B")
        self.style.configure("HeaderSub.TLabel", font=("Segoe UI", 9), foreground="#94A3B8", background="#1E293B")
        self.style.configure("Action.TButton", font=("Segoe UI", 10, "bold"), padding=6)
        self.style.configure("Secondary.TButton", font=("Segoe UI", 9), padding=5)
        self.style.configure("Status.TLabel", font=("Segoe UI", 9), foreground="#64748B", background="#F1F5F9")

    def _build_ui(self):
        # 1. Header Bar
        header_frame = ttk.Frame(self.root, style="Header.TFrame", padding=14)
        header_frame.pack(fill=tk.X)

        title_lbl = ttk.Label(header_frame, text="Adaptive-RAG Assistant", style="HeaderTitle.TLabel")
        title_lbl.pack(anchor=tk.W)

        subtitle_lbl = ttk.Label(
            header_frame,
            text="Dynamic Knowledge Routing (Qwen2.5-1.5B · BGE-small · FAISS · Layer-1 Contextual Routing)",
            style="HeaderSub.TLabel"
        )
        subtitle_lbl.pack(anchor=tk.W, pady=(2, 0))

        # 2. Conversation Display
        chat_container = ttk.Frame(self.root, padding=(12, 10, 12, 6))
        chat_container.pack(fill=tk.BOTH, expand=True)

        self.chat_history = scrolledtext.ScrolledText(
            chat_container,
            wrap=tk.WORD,
            font=("Segoe UI", 10),
            bg="#FFFFFF",
            fg="#0F172A",
            relief=tk.FLAT,
            padx=12,
            pady=12,
            state=tk.DISABLED
        )
        self.chat_history.pack(fill=tk.BOTH, expand=True)

        # Configure formatting tags
        self.chat_history.tag_configure("user_header", font=("Segoe UI", 10, "bold"), foreground="#2563EB")
        self.chat_history.tag_configure("user_msg", font=("Segoe UI", 10), foreground="#1E293B", lmargin1=8, lmargin2=8)
        self.chat_history.tag_configure("bot_header", font=("Segoe UI", 10, "bold"), foreground="#059669")
        self.chat_history.tag_configure("bot_msg", font=("Segoe UI", 10), foreground="#0F172A", lmargin1=8, lmargin2=8)
        self.chat_history.tag_configure("routing_badge_direct", font=("Segoe UI", 9, "bold"), foreground="#1D4ED8", background="#DBEAFE")
        self.chat_history.tag_configure("routing_badge_rag", font=("Segoe UI", 9, "bold"), foreground="#047857", background="#D1FAE5")
        self.chat_history.tag_configure("meta_info", font=("Segoe UI", 8), foreground="#64748B")
        self.chat_history.tag_configure("separator", font=("Segoe UI", 4), foreground="#E2E8F0")

        # 3. Input Controls Frame
        input_container = ttk.Frame(self.root, padding=(12, 6, 12, 6))
        input_container.pack(fill=tk.X)

        self.entry_var = tk.StringVar()
        self.input_entry = ttk.Entry(input_container, textvariable=self.entry_var, font=("Segoe UI", 11))
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8), ipady=5)
        self.input_entry.bind("<Return>", lambda event: self.send_message())

        self.send_btn = ttk.Button(input_container, text="Send", style="Action.TButton", command=self.send_message)
        self.send_btn.pack(side=tk.LEFT, padx=(0, 6))

        self.clear_btn = ttk.Button(input_container, text="Clear Chat", style="Secondary.TButton", command=self.clear_chat)
        self.clear_btn.pack(side=tk.LEFT)

        # 4. Status Bar
        self.status_var = tk.StringVar(value="Initializing Adaptive-RAG backend (loading Qwen on CUDA)...")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, style="Status.TLabel", padding=(14, 4))
        status_bar.pack(anchor=tk.W)

    def _start_background_init(self):
        def _init():
            try:
                self.pipeline = AdaptiveRAGPipeline(mode="adaptive")
                self.root.after(0, self._on_init_success)
            except Exception as e:
                self.root.after(0, lambda: self._on_init_fail(str(e)))
        threading.Thread(target=_init, daemon=True).start()

    def _on_init_success(self):
        self.status_var.set("Ready. Model loaded on CUDA (GTX 1650) with Layer-1 classifier.")
        self._append_message(
            "Adaptive-RAG",
            "Hello! I am your Adaptive-RAG assistant. I automatically evaluate whether your question requires external retrieval or can be answered directly from parametric memory.\nAsk me any question!",
            meta="System initialized"
        )
        self.input_entry.focus_set()

    def _on_init_fail(self, err_msg):
        self.status_var.set(f"Backend Initialization Error: {err_msg}")
        self._append_message("System", f"Failed to initialize Adaptive-RAG backend: {err_msg}")

    def send_message(self):
        query = self.entry_var.get().strip()
        if not query or self.is_generating:
            return

        if self.pipeline is None:
            self.status_var.set("Please wait, backend is still initializing...")
            return

        # Display user message immediately
        self._append_message("You", query)
        self.entry_var.set("")
        self.is_generating = True
        self.send_btn.configure(state=tk.DISABLED)
        self.status_var.set("Analyzing query with Layer-1 extractor & generating answer...")

        # Run inference in background thread to prevent UI freezing
        def _worker():
            try:
                start_t = time.perf_counter()
                res = self.pipeline.answer(query)
                elapsed = time.perf_counter() - start_t
                self.root.after(0, lambda: self._on_answer_ready(res, elapsed))
            except Exception as e:
                self.root.after(0, lambda: self._on_answer_error(str(e)))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_answer_ready(self, res, elapsed_s):
        self.is_generating = False
        self.send_btn.configure(state=tk.NORMAL)
        self.status_var.set("Ready")

        decision = res.get("routing_decision", "DIRECT")
        prob_l1 = res.get("classifier_probability", 0.0)
        tot_lat_ms = res.get("total_latency_ms", elapsed_s * 1000)

        meta = f"Routing: {decision} | P(RAG)={prob_l1*100:.1f}% | Latency: {tot_lat_ms:.0f} ms"
        self._append_message("Adaptive-RAG", res.get("answer", ""), routing=decision, meta=meta)
        self.input_entry.focus_set()

    def _on_answer_error(self, err_msg):
        self.is_generating = False
        self.send_btn.configure(state=tk.NORMAL)
        self.status_var.set(f"Error: {err_msg}")
        self._append_message("System", f"Error generating answer: {err_msg}")

    def _append_message(self, sender, text, routing=None, meta=None):
        self.chat_history.configure(state=tk.NORMAL)
        if sender == "You":
            self.chat_history.insert(tk.END, "You\n", "user_header")
            self.chat_history.insert(tk.END, f"{text}\n\n", "user_msg")
        elif sender == "Adaptive-RAG":
            self.chat_history.insert(tk.END, "Adaptive-RAG ", "bot_header")
            if routing == "RAG":
                self.chat_history.insert(tk.END, " [RAG RETRIEVAL] ", "routing_badge_rag")
            elif routing == "DIRECT":
                self.chat_history.insert(tk.END, " [DIRECT LLM] ", "routing_badge_direct")
            self.chat_history.insert(tk.END, "\n")
            self.chat_history.insert(tk.END, f"{text}\n", "bot_msg")
            if meta:
                self.chat_history.insert(tk.END, f"{meta}\n", "meta_info")
            self.chat_history.insert(tk.END, "\n")
        else:
            self.chat_history.insert(tk.END, f"[{sender}]: {text}\n\n", "meta_info")

        self.chat_history.configure(state=tk.DISABLED)
        self.chat_history.see(tk.END)

    def clear_chat(self):
        self.chat_history.configure(state=tk.NORMAL)
        self.chat_history.delete("1.0", tk.END)
        self.chat_history.configure(state=tk.DISABLED)
        self._append_message("System", "Chat history cleared.")


def main():
    parser = argparse.ArgumentParser(description="Adaptive-RAG Normal Chat Application")
    parser.add_argument("--test", action="store_true", help="Run automated headless self-test")
    parser.add_argument("--query", type=str, default=None, help="Process single query via CLI")
    args = parser.parse_args()

    if args.test or args.query:
        print("=== RUNNING CHAT CLI / TEST MODE ===")
        pipeline = AdaptiveRAGPipeline(mode="adaptive")
        test_q = args.query or "What is the capital of France?"
        print(f"Question: \"{test_q}\"")
        res = pipeline.answer(test_q)
        print(f"Routing Decision: {res['routing_decision']} (P(RAG)={res['classifier_probability']})")
        print(f"Answer:           {res['answer']}")
        print(f"Total Latency:    {res['total_latency_ms']:.2f} ms")
        assert len(res["answer"].strip()) > 0, "Test query produced empty answer"
        print("✓ Chat test successfully completed!")
        sys.exit(0)

    # Launch Tkinter GUI
    root = tk.Tk()
    app = AdaptiveChatApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
