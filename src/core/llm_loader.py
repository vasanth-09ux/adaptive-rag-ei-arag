"""
LLM Loader and Inference Manager for Qwen2.5-1.5B-Instruct.
Loads the model onto CUDA (fp16) once and provides inference helpers.
"""

from typing import List, Dict, Any, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config.config import config, LLMConfig


class LLMManager:
    """Manages Qwen model lifecycle and text generation on GPU."""

    _instance: Optional["LLMManager"] = None

    def __init__(self, cfg: LLMConfig = config.llm):
        self.cfg = cfg
        self.device = self.cfg.device
        self.dtype = self.cfg.dtype

        print(f"Loading tokenizer for {self.cfg.model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_id)

        print(f"Loading model {self.cfg.model_id} on {self.device} with {self.dtype}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_id,
            dtype=self.dtype
        ).to(self.device)
        self.model.eval()

        print("LLM loaded successfully.")

    @classmethod
    def get_instance(cls, cfg: LLMConfig = config.llm) -> "LLMManager":
        """Singleton accessor to prevent reloading the LLM into GPU memory."""
        if cls._instance is None:
            cls._instance = cls(cfg)
        return cls._instance

    def generate(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> str:
        """
        Generates text using the Qwen chat template.
        """
        tokens_limit = max_new_tokens if max_new_tokens is not None else self.cfg.max_new_tokens
        temp = temperature if temperature is not None else self.cfg.temperature

        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer([prompt_text], return_tensors="pt").to(self.device)

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": tokens_limit,
            "do_sample": False if temp == 0.0 else True,
            "pad_token_id": self.tokenizer.eos_token_id
        }
        if temp > 0.0:
            gen_kwargs["temperature"] = temp

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        input_len = inputs.input_ids.shape[1]
        new_tokens = output_ids[0][input_len:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        return response
