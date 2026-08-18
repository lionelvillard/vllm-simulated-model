import math
import types

import pytest

from vllm_simulated.latency import BatchShape
from vllm_simulated.physics_latency import PhysicsLatencyModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hf(
    num_hidden_layers=1,
    hidden_size=64,
    num_attention_heads=4,
    num_key_value_heads=4,
    intermediate_size=128,
    **kwargs,
):
    return types.SimpleNamespace(
        num_hidden_layers=num_hidden_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        intermediate_size=intermediate_size,
        **kwargs,
    )


def _config(peak_tflops=1.0, hbm_gbps=1.0, weight_dtype="bfloat16",
            beta=None, tp=1):
    return {
        "hardware": {
            "peak_tflops": peak_tflops,
            "hbm_gbps": hbm_gbps,
            "weight_dtype": weight_dtype,
        },
        "beta": beta or [1.0, 1.0, 0.0],
        "tp": tp,
    }


def _shape(
    num_prefill_tokens=0,
    num_decode_seqs=0,
    sum_context_len=0,
    num_prefill_seqs=0,
    sum_decode_context_len=0,
):
    return BatchShape(
        num_prefill_tokens=num_prefill_tokens,
        num_decode_seqs=num_decode_seqs,
        sum_context_len=sum_context_len,
        num_prefill_seqs=num_prefill_seqs,
        sum_decode_context_len=sum_decode_context_len,
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_construction_minimal():
    model = PhysicsLatencyModel.from_dict(_config(), hf_config=_hf())
    assert model is not None


def test_construction_missing_hardware():
    with pytest.raises(ValueError, match="hardware"):
        PhysicsLatencyModel.from_dict({}, hf_config=_hf())


def test_construction_unknown_dtype():
    with pytest.raises(ValueError, match="weight_dtype"):
        PhysicsLatencyModel.from_dict(
            _config(weight_dtype="float32"), hf_config=_hf()
        )


def test_construction_wrong_beta_length():
    cfg = _config()
    cfg["beta"] = [1.0, 1.0]  # only 2 values
    with pytest.raises(ValueError, match="beta"):
        PhysicsLatencyModel.from_dict(cfg, hf_config=_hf())


def test_construction_negative_beta():
    cfg = _config()
    cfg["beta"] = [1.0, -0.1, 0.0]
    with pytest.raises(ValueError, match="beta"):
        PhysicsLatencyModel.from_dict(cfg, hf_config=_hf())


# ---------------------------------------------------------------------------
# Empty batch
# ---------------------------------------------------------------------------

def test_empty_batch_returns_beta_base():
    cfg = _config(beta=[1.0, 1.0, 5.0])
    model = PhysicsLatencyModel.from_dict(cfg, hf_config=_hf())
    result = model.step_time_ms(_shape())
    assert math.isclose(result, 5.0, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# Prefill — compute-bound vs KV-write-bound
# ---------------------------------------------------------------------------
# Model: L=1, d=64, h=4, dH=16, dKV_full=64, dFF=128
# Compute-bound HW: peak_tflops=0.001 (slow), hbm_gbps=1e9 (fast memory)
#   peak_flops_ms = 0.001 * 1e9 = 1e6 FLOPs/ms
#   hbm_bw_ms    = 1e9 * 1e6   = 1e15 bytes/ms  (effectively infinite)
# KV-write-bound HW: peak_tflops=1e9 (fast), hbm_gbps=0.001 (slow memory)
#   peak_flops_ms = 1e9 * 1e9 = 1e18 FLOPs/ms   (effectively infinite)
#   hbm_bw_ms    = 0.001 * 1e6 = 1e3 bytes/ms
#
# Batch: N_pf=10 tokens, 2 prefill seqs, sum_pf_ctx=20, avg_pf_ctx=10
#   (sum_context_len=20, sum_decode_context_len=0, num_prefill_seqs=2)
#
# T_pf_compute (tp=1):
#   proj  = 1 * 2 * 10 * 64 * (128 + 128) = 1 * 2 * 10 * 64 * 256 = 327680
#   attn  = 1 * 4 * 4 * 10 * 10 * 16     = 1 * 4 * 4 * 10 * 10 * 16 = 25600
#   ffn   = 1 * 6 * 64 * 128 * 10         = 491520
#   total = 844800 FLOPs
#   T_compute = 844800 / 1e6 = 0.8448 ms
#
# T_pf_kv:
#   = 1 * 2 * 64 * 10 * 2 / bw = 2560 / bw
#   compute-bound: 2560 / 1e15 ≈ 0 ms  → compute wins
#   kv-bound:      2560 / 1e3  = 2.56 ms → kv wins

def test_prefill_compute_bound():
    # Slow compute, fast memory → T_compute > T_kv
    model = PhysicsLatencyModel.from_dict(
        _config(peak_tflops=0.001, hbm_gbps=1e9, beta=[1.0, 1.0, 0.0]),
        hf_config=_hf(),
    )
    shape = _shape(
        num_prefill_tokens=10, num_prefill_seqs=2,
        sum_context_len=20, sum_decode_context_len=0,
    )
    result = model.step_time_ms(shape)
    # T_prefill ≈ T_pf_compute = 0.8448 ms; T_kv negligible; T_decode=0
    assert math.isclose(result, 0.8448, rel_tol=1e-4)


def test_prefill_kv_write_bound():
    # Fast compute, slow memory → T_kv > T_compute
    model = PhysicsLatencyModel.from_dict(
        _config(peak_tflops=1e9, hbm_gbps=0.001, beta=[1.0, 1.0, 0.0]),
        hf_config=_hf(),
    )
    shape = _shape(
        num_prefill_tokens=10, num_prefill_seqs=2,
        sum_context_len=20, sum_decode_context_len=0,
    )
    result = model.step_time_ms(shape)
    # T_pf_kv = 2560 / 1e3 = 2.56 ms; T_compute negligible
    assert math.isclose(result, 2.56, rel_tol=1e-4)


def test_prefill_zero_tokens_is_zero():
    model = PhysicsLatencyModel.from_dict(_config(), hf_config=_hf())
    shape = _shape(num_prefill_tokens=0)
    # Prefill term must be 0 regardless of sums
    result_decode_only = model.step_time_ms(
        _shape(num_decode_seqs=1, sum_context_len=10, sum_decode_context_len=10)
    )
    result_no_pf = model.step_time_ms(shape)
    assert result_no_pf >= 0.0
    # No prefill contribution when N_pf=0
    result_with_pf = model.step_time_ms(
        _shape(
            num_prefill_tokens=10, num_prefill_seqs=1,
            sum_context_len=10, sum_decode_context_len=0,
        )
    )
    assert result_with_pf > result_no_pf


# ---------------------------------------------------------------------------
# Decode — compute-bound vs weight-load+KV-bound
# ---------------------------------------------------------------------------
# Batch: N_dc=1, sum_dc_ctx=100
#
# T_dc_compute (fast memory, slow compute: peak=0.001, hbm=1e9):
#   proj  = 1 * 2 * 1 * 64 * 256 = 32768
#   attn  = 1 * 4 * 4 * 100 * 16 = 25600
#   ffn   = 1 * 6 * 64 * 128 * 1 = 49152
#   total = 107520 FLOPs
#   T_dc_compute = 107520 / 1e6 = 0.10752 ms
#
# T_weight + T_dc_kv (fast compute, slow memory: peak=1e9, hbm=0.001):
#   w_attn  = 1 * 64 * 256 * 2 = 32768 bytes
#   w_dense = 1 * 3 * 64 * 128 * 2 = 49152 bytes
#   T_weight = 81920 / 1e3 = 81.92 ms
#   T_dc_kv  = 1 * 2 * 64 * 101 * 2 / 1e3 = 25856 / 1e3 = 25.856 ms
#   T_weight + T_dc_kv = 107.776 ms

def test_decode_compute_bound():
    # Slow compute → T_compute > T_weight + T_kv
    model = PhysicsLatencyModel.from_dict(
        _config(peak_tflops=0.001, hbm_gbps=1e9, beta=[1.0, 1.0, 0.0]),
        hf_config=_hf(),
    )
    shape = _shape(
        num_decode_seqs=1, sum_context_len=100, sum_decode_context_len=100,
    )
    result = model.step_time_ms(shape)
    assert math.isclose(result, 0.10752, rel_tol=1e-4)


def test_decode_weight_and_kv_bound():
    # Slow memory → T_weight + T_kv > T_compute
    model = PhysicsLatencyModel.from_dict(
        _config(peak_tflops=1e9, hbm_gbps=0.001, beta=[1.0, 1.0, 0.0]),
        hf_config=_hf(),
    )
    shape = _shape(
        num_decode_seqs=1, sum_context_len=100, sum_decode_context_len=100,
    )
    result = model.step_time_ms(shape)
    assert math.isclose(result, 107.776, rel_tol=1e-4)


def test_decode_zero_seqs_is_zero():
    model = PhysicsLatencyModel.from_dict(_config(), hf_config=_hf())
    shape = _shape(num_decode_seqs=0)
    result = model.step_time_ms(shape)
    assert result == 0.0  # beta_base is also 0 with default [1,1,0]


# ---------------------------------------------------------------------------
# Betas
# ---------------------------------------------------------------------------

def test_beta_pf_scales_prefill():
    # beta_pf=2 should double T_prefill relative to beta_pf=1
    shape = _shape(
        num_prefill_tokens=10, num_prefill_seqs=2,
        sum_context_len=20, sum_decode_context_len=0,
    )
    model1 = PhysicsLatencyModel.from_dict(
        _config(beta=[1.0, 1.0, 0.0]), hf_config=_hf()
    )
    model2 = PhysicsLatencyModel.from_dict(
        _config(beta=[2.0, 1.0, 0.0]), hf_config=_hf()
    )
    assert math.isclose(
        model2.step_time_ms(shape), 2.0 * model1.step_time_ms(shape), rel_tol=1e-9
    )


def test_beta_dc_scales_decode():
    shape = _shape(
        num_decode_seqs=1, sum_context_len=100, sum_decode_context_len=100,
    )
    model1 = PhysicsLatencyModel.from_dict(
        _config(beta=[1.0, 1.0, 0.0]), hf_config=_hf()
    )
    model2 = PhysicsLatencyModel.from_dict(
        _config(beta=[1.0, 3.0, 0.0]), hf_config=_hf()
    )
    assert math.isclose(
        model2.step_time_ms(shape), 3.0 * model1.step_time_ms(shape), rel_tol=1e-9
    )


def test_beta_base_adds_to_any_batch():
    shape = _shape(num_prefill_tokens=5, num_prefill_seqs=1, sum_context_len=5)
    model1 = PhysicsLatencyModel.from_dict(
        _config(beta=[1.0, 1.0, 0.0]), hf_config=_hf()
    )
    model2 = PhysicsLatencyModel.from_dict(
        _config(beta=[1.0, 1.0, 7.5]), hf_config=_hf()
    )
    assert math.isclose(
        model2.step_time_ms(shape) - model1.step_time_ms(shape), 7.5, rel_tol=1e-9
    )


# ---------------------------------------------------------------------------
# Mixed batch
# ---------------------------------------------------------------------------

def test_mixed_batch_greater_than_either_alone():
    hf = _hf()
    model = PhysicsLatencyModel.from_dict(_config(), hf_config=hf)
    pf_only = _shape(
        num_prefill_tokens=10, num_prefill_seqs=1,
        sum_context_len=10, sum_decode_context_len=0,
    )
    dc_only = _shape(
        num_decode_seqs=2, sum_context_len=50, sum_decode_context_len=50,
    )
    mixed = _shape(
        num_prefill_tokens=10, num_prefill_seqs=1,
        num_decode_seqs=2,
        sum_context_len=60, sum_decode_context_len=50,
    )
    assert model.step_time_ms(mixed) > model.step_time_ms(pf_only)
    assert model.step_time_ms(mixed) > model.step_time_ms(dc_only)


# ---------------------------------------------------------------------------
# MoE — weight and FLOPs accounting
# ---------------------------------------------------------------------------
# MoE model: L=2 layers (all MoE), d=64, dH=16, num_heads=4, num_kv_heads=4,
#   dKV_full=64, num_experts=4, kEff=2, dFFMoE=128, shared_ffn=0, dFF=0
# Dense baseline: same but no MoE, intermediate_size=128 (2 * kEff/numExperts * dFFMoE)
#   → dense effectively has same activated FFN compute but different weight footprint


def _moe_hf(**kwargs):
    return _hf(
        num_hidden_layers=2,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=128,
        **{"intermediate_size": 0, **kwargs},
    )


def test_moe_higher_weight_than_dense_equivalent():
    # MoE loads kEff=2 expert weights per step.
    # Dense baseline with intermediate_size=64 (same compute as
    # kEff/numExperts fraction).
    # At large batch with slow memory, MoE T_weight > dense T_weight.
    shape = _shape(
        num_decode_seqs=1, sum_context_len=10, sum_decode_context_len=10,
    )
    moe_model = PhysicsLatencyModel.from_dict(
        _config(peak_tflops=1e9, hbm_gbps=0.001),
        hf_config=_moe_hf(),
    )
    # Dense with same arch but dFF=64 (kEff/numExperts * dFFMoE = 2/4 * 128 = 64)
    dense_model = PhysicsLatencyModel.from_dict(
        _config(peak_tflops=1e9, hbm_gbps=0.001),
        hf_config=_hf(num_hidden_layers=2, intermediate_size=64),
    )
    assert moe_model.step_time_ms(shape) > dense_model.step_time_ms(shape)


def test_moe_layer_split_interleaved():
    # decoder_sparse_step=1 → every other layer is MoE → num_moe_layers = L/2
    # With L=4: 2 MoE + 2 dense layers
    hf = _hf(
        num_hidden_layers=4,
        intermediate_size=128,    # dense FFN
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=128,
        decoder_sparse_step=1,    # interleave step: 1 → every 2nd layer
    )
    model = PhysicsLatencyModel.from_dict(_config(), hf_config=hf)
    # Verify model was constructed without error (layer split logic ran)
    assert model is not None
    # With interleaved MoE, both dense and MoE layers contribute.
    # step_time_ms should be positive for a non-empty decode batch.
    shape = _shape(num_decode_seqs=1, sum_context_len=10, sum_decode_context_len=10)
    assert model.step_time_ms(shape) > 0.0


def test_moe_shared_expert_adds_to_flops():
    # Shared expert increases FLOPs beyond routed experts alone.
    shape = _shape(
        num_prefill_tokens=10, num_prefill_seqs=1,
        sum_context_len=10, sum_decode_context_len=0,
    )
    model_no_shared = PhysicsLatencyModel.from_dict(
        _config(peak_tflops=0.001, hbm_gbps=1e9),
        hf_config=_moe_hf(),
    )
    model_with_shared = PhysicsLatencyModel.from_dict(
        _config(peak_tflops=0.001, hbm_gbps=1e9),
        hf_config=_moe_hf(shared_expert_intermediate_size=32),
    )
    assert model_with_shared.step_time_ms(shape) > model_no_shared.step_time_ms(shape)


def test_all_moe_layers_when_no_interleave():
    # When decoder_sparse_step is absent (or 0) and num_experts > 0, all layers are
    # MoE (num_dense_layers == 0).  intermediate_size (the dense FFN size) is
    # irrelevant because there are no dense FFN layers — the model only uses
    # expert FLOPs/weights.  Verified by showing that setting intermediate_size=64
    # produces identical cost to intermediate_size=0 in a pure-MoE (no interleave)
    # model.
    shape = _shape(num_decode_seqs=1, sum_context_len=10, sum_decode_context_len=10)
    moe_no_dense = PhysicsLatencyModel.from_dict(
        _config(peak_tflops=1e9, hbm_gbps=0.001),
        hf_config=_moe_hf(),  # no decoder_sparse_step, intermediate_size=0
    )
    moe_with_unused_ffn = PhysicsLatencyModel.from_dict(
        _config(peak_tflops=1e9, hbm_gbps=0.001),
        hf_config=_moe_hf(intermediate_size=64),  # set but irrelevant
    )
    # num_dense_layers == 0 → intermediate_size has no effect on weight or FLOPs
    assert math.isclose(
        moe_no_dense.step_time_ms(shape),
        moe_with_unused_ffn.step_time_ms(shape),
        rel_tol=1e-9,
    )
