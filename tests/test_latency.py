import math
from types import SimpleNamespace

import pytest
import torch

from vllm_simulated.latency import (
    BatchShape,
    LatencyConfig,
    SimulatedLatencyModel,
    batch_shape_from_attn_metadata,
)


def test_step_time_is_linear_combination():
    cfg = LatencyConfig(
        base_ms=5.0,
        prefill_ms_per_token=0.05,
        decode_ms_per_seq=1.2,
        ctx_ms_per_ktoken=0.3,
    )
    model = SimulatedLatencyModel(cfg)
    shape = BatchShape(
        num_prefill_tokens=100, num_decode_seqs=10, sum_context_len=2000
    )
    # 5 + 0.05*100 + 1.2*10 + 0.3*(2000/1000) = 5 + 5 + 12 + 0.6 = 22.6
    assert math.isclose(model.step_time_ms(shape), 22.6, rel_tol=1e-9)


def test_step_time_never_negative():
    model = SimulatedLatencyModel(LatencyConfig())
    assert model.step_time_ms(BatchShape(0, 0, 0)) == 0.0


def test_from_dict_rejects_unknown_keys():
    with pytest.raises(ValueError, match="Unknown latency config keys"):
        LatencyConfig.from_dict({"base_ms": 1.0, "bogus": 2.0})


def test_from_dict_rejects_negative_coefficients():
    with pytest.raises(ValueError, match="must be >= 0"):
        LatencyConfig.from_dict({"decode_ms_per_seq": -1.0})


def test_batch_shape_from_attn_metadata():
    # 3 requests with query lengths 1, 1, 3 -> 2 decodes, 1 prefill of 3 tokens.
    md = SimpleNamespace(
        query_start_loc=torch.tensor([0, 1, 2, 5]),
        seq_lens=torch.tensor([10, 12, 7]),
    )
    shape = batch_shape_from_attn_metadata(md)
    assert shape == BatchShape(
        num_prefill_tokens=3, num_decode_seqs=2, sum_context_len=29
    )
