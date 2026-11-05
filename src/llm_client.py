"""Provider-agnostic LLM Client interface with Kaggle T4 x2 vLLM support."""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type
from pydantic import BaseModel


class LLMClient(ABC):
    """Abstract base interface for structured JSON generation."""

    @abstractmethod
    def generate_json(
        self,
        schema: Type[BaseModel] | Dict[str, Any],
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        timeout: float = 30.0,
    ) -> Optional[Dict[str, Any]]:
        raise NotImplementedError


class MockLLMClient(LLMClient):
    """Deterministic mock client for unit and integration testing."""

    def __init__(self, canned_responses: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        self.canned_responses = canned_responses or {}
        self.call_count = 0

    def generate_json(
        self,
        schema: Type[BaseModel] | Dict[str, Any],
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        timeout: float = 30.0,
    ) -> Optional[Dict[str, Any]]:
        self.call_count += 1
        for key, resp in self.canned_responses.items():
            if key in user_prompt:
                return resp
        return None


class KaggleT4x2VLLMClient(LLMClient):
    """Local vLLM client tuned specifically for Kaggle 2x Tesla T4 (32GB total VRAM, SM75).

    Key hardware settings for T4 x2:
    - `tensor_parallel_size=2`: shards model weights across both T4 GPUs.
    - `dtype="float16"`: T4 (Turing SM75) does NOT support native bfloat16; float16 is mandatory.
    - `max_model_len=8192`, `gpu_memory_utilization=0.90`.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-Coder-14B-Instruct-AWQ",
        tensor_parallel_size: int = 2,
        dtype: str = "float16",
        max_model_len: int = 8192,
        gpu_memory_utilization: float = 0.90,
    ) -> None:
        self.model_name = model_name
        self.tensor_parallel_size = tensor_parallel_size
        self.dtype = dtype
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self._llm: Any = None
        self._sampling_params_cls: Any = None

    def _lazy_init(self) -> None:
        if self._llm is not None:
            return
        from vllm import LLM, SamplingParams  # type: ignore

        self._sampling_params_cls = SamplingParams
        self._llm = LLM(
            model=self.model_name,
            tensor_parallel_size=self.tensor_parallel_size,
            dtype=self.dtype,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
            trust_remote_code=True,
        )

    def generate_json(
        self,
        schema: Type[BaseModel] | Dict[str, Any],
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        timeout: float = 30.0,
    ) -> Optional[Dict[str, Any]]:
        self._lazy_init()
        prompt = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        params = self._sampling_params_cls(
            temperature=temperature,
            max_tokens=1024,
        )
        outputs = self._llm.generate([prompt], params, use_tqdm=False)
        if not outputs or not outputs[0].outputs:
            return None
        text = outputs[0].outputs[0].text.strip()
        # Strip accidental markdown code fences if present
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:]).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None
