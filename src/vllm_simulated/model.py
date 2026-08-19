import time

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention import Attention

from vllm_simulated.latency import (
    LatencyModel,
    batch_shape_from_attn_metadata,
    build_latency_model,
)


def _build_latency_model(hf_config) -> LatencyModel:
    """Build a LatencyModel from hf_config, raising on missing latency block."""
    latency = getattr(hf_config, "latency", None)
    if not latency:
        raise ValueError(
            "SimulatedForCausalLM requires a non-empty 'latency' block in "
            "the model config; none was found. See the plugin README."
        )
    return build_latency_model(latency, hf_config=hf_config)


class SimulatedForCausalLM(nn.Module):
    """Simulated causal LM that emits random tokens and sleeps per step."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config
        hf_config = model_config.hf_config

        self.hidden_size = model_config.get_hidden_size()
        self.vocab_size = model_config.get_vocab_size()
        self.dtype = model_config.dtype
        num_layers = model_config.get_total_num_hidden_layers()
        num_heads = model_config.get_num_attention_heads(parallel_config)
        num_kv_heads = model_config.get_num_kv_heads(parallel_config)
        head_size = model_config.get_head_size()

        self.attn_layers = nn.ModuleList(
            [
                Attention(
                    num_heads=num_heads,
                    head_size=head_size,
                    scale=head_size**-0.5,
                    num_kv_heads=num_kv_heads,
                    cache_config=vllm_config.cache_config,
                    quant_config=vllm_config.quant_config,
                    prefix=f"{prefix}.layers.{i}.attn",
                )
                for i in range(num_layers)
            ]
        )

        self.latency = _build_latency_model(hf_config)
        self.deterministic_length = getattr(hf_config, "latency", {}).get(
            "deterministic_length", True
        )
        self.eos_token_id = getattr(hf_config, "eos_token_id", None)

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return torch.zeros(
            input_ids.shape[0], self.hidden_size, dtype=self.dtype
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            num_tokens = inputs_embeds.shape[0]
        else:
            num_tokens = input_ids.shape[0]

        md = self._current_attn_metadata()
        if md is not None:
            shape = batch_shape_from_attn_metadata(md)
            sleep_s = self.latency.step_time_ms(shape) / 1000.0
            if sleep_s > 0:
                time.sleep(sleep_s)

        return torch.zeros(num_tokens, self.hidden_size, dtype=self.dtype)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]
        logits = torch.randn(num_tokens, self.vocab_size)
        if self.deterministic_length and self.eos_token_id is not None:
            logits[:, self.eos_token_id] = float("-inf")
        return logits

    def load_weights(self, weights) -> set[str]:
        for _ in weights:
            pass
        return set()

    @staticmethod
    def _current_attn_metadata():
        attn_metadata = get_forward_context().attn_metadata
        if not attn_metadata:
            return None
        if isinstance(attn_metadata, list):
            attn_metadata = attn_metadata[0]
        return next(iter(attn_metadata.values()), None)
