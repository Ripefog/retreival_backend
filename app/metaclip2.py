"""Inference wrapper for MetaCLIP 2 text embeddings used by Milvus."""

from __future__ import annotations

from typing import Any
import inspect

import numpy as np
import torch
import torch.nn.functional as F

# Transformers 4.56 uses the public pytree API, introduced after Torch 2.1.
# Keep the MetaCLIP 2 runtime compatible with the pinned CUDA 11.8 image.
import torch.utils._pytree as _torch_pytree

if not hasattr(_torch_pytree, "register_pytree_node") and hasattr(_torch_pytree, "_register_pytree_node"):
    _legacy_register = _torch_pytree._register_pytree_node
    _legacy_parameters = set(inspect.signature(_legacy_register).parameters)

    def _register_pytree_node(*args, **kwargs):
        compatible_kwargs = {key: value for key, value in kwargs.items() if key in _legacy_parameters}
        return _legacy_register(*args, **compatible_kwargs)

    _torch_pytree.register_pytree_node = _register_pytree_node

from transformers import AutoModel, AutoProcessor


class MetaCLIP2Encoder:
    """Loads MetaCLIP 2 and returns normalized float32 embeddings."""

    def __init__(self, model_name: str, device: str, expected_dim: int, cache_dir: str | None = None):
        self.device = torch.device(device)
        self.expected_dim = expected_dim
        self.dtype = self._select_dtype()
        self.processor = AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir, use_fast=False)
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            cache_dir=cache_dir,
        ).to(self.device).eval()
        self.output_dim = int(self.model.config.projection_dim)
        if self.output_dim != expected_dim:
            raise ValueError(
                f"MetaCLIP 2 projection dimension is {self.output_dim}, expected {expected_dim}."
            )

    def _select_dtype(self) -> torch.dtype:
        if self.device.type != "cuda":
            return torch.float32
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16

    @staticmethod
    def _features(outputs: Any) -> torch.Tensor:
        if isinstance(outputs, torch.Tensor):
            features = outputs
        else:
            features = getattr(outputs, "pooler_output", None)
            if features is None and isinstance(outputs, dict):
                features = outputs.get("pooler_output")
        if not isinstance(features, torch.Tensor) or features.ndim != 2:
            raise TypeError("MetaCLIP 2 did not return batched projected features.")
        return F.normalize(features.float(), p=2, dim=-1)

    def encode_text(self, text: str) -> np.ndarray:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Text must be a non-empty string.")
        inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True)
        model_inputs = {
            key: value.to(self.device, non_blocking=True)
            for key, value in inputs.items()
            if key in {"input_ids", "attention_mask", "position_ids"}
        }
        with torch.inference_mode():
            features = self._features(self.model.get_text_features(**model_inputs))
        vector = features[0].cpu().numpy().astype(np.float32, copy=False)
        if vector.shape != (self.output_dim,):
            raise RuntimeError(f"Unexpected MetaCLIP 2 text embedding shape: {vector.shape}")
        return vector
