import types

import pytest
import torch

pytest.importorskip("vllm")

from vllm_simulated.model import (
    SimulatedForCausalLM,
    _load_latency_config,
)


def test_missing_latency_block_raises():
    with pytest.raises(ValueError, match="latency"):
        _load_latency_config(types.SimpleNamespace())
    with pytest.raises(ValueError, match="latency"):
        _load_latency_config(types.SimpleNamespace(latency={}))


def test_eos_masking_toggle():
    model = SimulatedForCausalLM.__new__(SimulatedForCausalLM)
    model.vocab_size = 10
    model.eos_token_id = 0
    hidden = torch.zeros(3, 4)

    model.deterministic_length = True
    masked = model.compute_logits(hidden)
    assert (masked[:, 0] == float("-inf")).all()

    model.deterministic_length = False
    unmasked = model.compute_logits(hidden)
    assert not torch.isinf(unmasked[:, 0]).any()
