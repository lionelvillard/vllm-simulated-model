from dataclasses import dataclass

_WEIGHT_DTYPE_BPP: dict[str, float] = {
    "bfloat16": 2.0,
    "float16": 2.0,
    "int8": 1.0,
    "float8": 1.0,
    "float8_e4m3fn": 1.0,
    "float8_e5m2": 1.0,
}

_ABPP = 2.0  # activations always bf16/fp16


@dataclass(frozen=True)
class PhysicsConfig:
    peak_tflops: float
    hbm_gbps: float
    weight_dtype: str
    beta: tuple
    tp: int
    deterministic_length: bool

    @property
    def peak_flops_ms(self) -> float:
        return self.peak_tflops * 1e9

    @property
    def hbm_bw_ms(self) -> float:
        return self.hbm_gbps * 1e6

    @property
    def wbpp(self) -> float:
        return _WEIGHT_DTYPE_BPP[self.weight_dtype]

    @classmethod
    def from_dict(cls, d: dict) -> "PhysicsConfig":
        hw = d.get("hardware")
        if not hw:
            raise ValueError(
                "physics latency model requires a 'hardware' block with "
                "peak_tflops, hbm_gbps, and weight_dtype"
            )
        peak_tflops = hw.get("peak_tflops")
        hbm_gbps = hw.get("hbm_gbps")
        weight_dtype = hw.get("weight_dtype", "bfloat16")
        if peak_tflops is None:
            raise ValueError("hardware.peak_tflops is required")
        if hbm_gbps is None:
            raise ValueError("hardware.hbm_gbps is required")
        if weight_dtype not in _WEIGHT_DTYPE_BPP:
            raise ValueError(
                f"Unknown weight_dtype: {weight_dtype!r}. "
                f"Known: {sorted(_WEIGHT_DTYPE_BPP)}"
            )
        beta_list = d.get("beta", [1.0, 1.0, 0.0])
        if len(beta_list) != 3:
            raise ValueError(
                f"beta must have exactly 3 values [beta_pf, beta_dc, beta_base], "
                f"got {len(beta_list)}"
            )
        if any(b < 0 for b in beta_list):
            raise ValueError("beta values must be >= 0")
        # tp is injected at load time from parallel_config.tensor_parallel_size
        # (see model.py); it is not a user-facing config.json key. The default
        # of 1 applies only when the model is built outside vLLM (e.g. tests).
        tp = d.get("tp", 1)
        if not isinstance(tp, int) or tp < 1:
            raise ValueError(f"tp must be a positive integer, got {tp!r}")
        det = d.get("deterministic_length", True)
        return cls(
            peak_tflops=float(peak_tflops),
            hbm_gbps=float(hbm_gbps),
            weight_dtype=weight_dtype,
            beta=tuple(float(b) for b in beta_list),
            tp=tp,
            deterministic_length=det,
        )


@dataclass(frozen=True)
class _ArchParams:
    L: int
    d: int
    dH: int
    h: int           # num_attention_heads (full, not per-GPU)
    dKV_full: int    # num_kv_heads * dH (full, for FLOPs formula)
    dFF: int         # intermediate_size (dense, full)
    num_moe_layers: int
    num_dense_layers: int
    kEff: int        # num_experts_per_tok (0 if dense)
    num_experts: int
    dFFMoE: int      # moe_intermediate_size (full, 0 if dense)
    shared_ffn: int  # shared_expert_intermediate_size (full, 0 if absent)

    @classmethod
    def from_hf_config(cls, hf_config, tp: int) -> "_ArchParams":
        L = getattr(hf_config, "num_hidden_layers", 0)
        d = getattr(hf_config, "hidden_size", 0)
        num_heads = getattr(hf_config, "num_attention_heads", 1)
        num_kv_heads = getattr(hf_config, "num_key_value_heads", num_heads)
        dH = getattr(hf_config, "head_dim", d // max(num_heads, 1))
        dKV_full = num_kv_heads * dH
        dFF = getattr(hf_config, "intermediate_size", 0)

        num_experts = getattr(hf_config, "num_experts", 0)
        kEff = getattr(hf_config, "num_experts_per_tok", 0)
        dFFMoE = getattr(hf_config, "moe_intermediate_size", 0) if num_experts else 0
        shared_ffn = getattr(hf_config, "shared_expert_intermediate_size", 0)

        interleave_step = getattr(hf_config, "decoder_sparse_step", 0)
        if num_experts > 0:
            if interleave_step > 0:
                num_moe_layers = L // (interleave_step + 1)
            else:
                num_moe_layers = L
        else:
            num_moe_layers = 0
        num_dense_layers = L - num_moe_layers

        return cls(
            L=L,
            d=d,
            dH=dH,
            h=num_heads,
            dKV_full=dKV_full,
            dFF=dFF,
            num_moe_layers=num_moe_layers,
            num_dense_layers=num_dense_layers,
            kEff=kEff,
            num_experts=num_experts,
            dFFMoE=dFFMoE,
            shared_ffn=shared_ffn,
        )


class PhysicsLatencyModel:
    def __init__(self, config: PhysicsConfig, arch: _ArchParams) -> None:
        self.config = config
        self.arch = arch

    def step_time_ms(self, shape) -> float:
        cfg = self.config
        arch = self.arch
        tp = cfg.tp

        N_pf = shape.num_prefill_tokens
        N_pf_seqs = max(shape.num_prefill_seqs, 1)
        N_dc = shape.num_decode_seqs
        sum_dc_ctx = shape.sum_decode_context_len
        sum_pf_ctx = shape.sum_context_len - sum_dc_ctx
        avg_pf_ctx = sum_pf_ctx / N_pf_seqs

        pf_ms = cfg.peak_flops_ms
        bw_ms = cfg.hbm_bw_ms
        beta_pf, beta_dc, beta_base = cfg.beta

        L = arch.L
        d = arch.d
        dH = arch.dH
        dKV_full = arch.dKV_full

        # ---- Prefill ----
        T_prefill = 0.0
        if N_pf > 0:
            proj_flops_pf = L * 2 * N_pf * d * (2 * d + 2 * dKV_full) / tp
            attn_flops_pf = L * 4 * (arch.h / tp) * N_pf * avg_pf_ctx * dH
            ffn_flops_pf = (
                arch.num_dense_layers * 6 * d * (arch.dFF / tp) * N_pf
                + arch.num_moe_layers * arch.kEff * 6 * d * (arch.dFFMoE / tp) * N_pf
                + arch.num_moe_layers * 6 * d * (arch.shared_ffn / tp) * N_pf
            )
            T_pf_compute = (proj_flops_pf + attn_flops_pf + ffn_flops_pf) / pf_ms
            T_pf_kv = L * 2 * (dKV_full / tp) * N_pf * _ABPP / bw_ms
            T_prefill = beta_pf * max(T_pf_compute, T_pf_kv)

        # ---- Decode ----
        T_decode = 0.0
        if N_dc > 0:
            proj_flops_dc = L * 2 * N_dc * d * (2 * d + 2 * dKV_full) / tp
            attn_flops_dc = L * 4 * (arch.h / tp) * sum_dc_ctx * dH
            ffn_flops_dc = (
                arch.num_dense_layers * 6 * d * (arch.dFF / tp) * N_dc
                + arch.num_moe_layers * arch.kEff * 6 * d * (arch.dFFMoE / tp) * N_dc
                + arch.num_moe_layers * 6 * d * (arch.shared_ffn / tp) * N_dc
            )
            T_dc_compute = (proj_flops_dc + attn_flops_dc + ffn_flops_dc) / pf_ms
            T_dc_kv = L * 2 * (dKV_full / tp) * (sum_dc_ctx + N_dc) * _ABPP / bw_ms
            w_attn = L * d * (2 * d + 2 * dKV_full) / tp * cfg.wbpp
            w_dense = arch.num_dense_layers * 3 * d * (arch.dFF / tp) * cfg.wbpp
            w_moe = (
                arch.num_moe_layers * arch.kEff * 3 * d
                * (arch.dFFMoE / tp) * cfg.wbpp
            )
            w_shared = arch.num_moe_layers * 3 * d * (arch.shared_ffn / tp) * cfg.wbpp
            T_weight = (w_attn + w_dense + w_moe + w_shared) / bw_ms
            T_decode = beta_dc * max(T_dc_compute, T_weight + T_dc_kv)

        return max(0.0, T_prefill + T_decode + beta_base)

    @classmethod
    def from_dict(cls, d: dict, hf_config=None, **kwargs) -> "PhysicsLatencyModel":
        config = PhysicsConfig.from_dict(d)
        if hf_config is None:
            raise ValueError(
                "PhysicsLatencyModel requires hf_config to derive architecture "
                "parameters (num_hidden_layers, hidden_size, etc.)"
            )
        arch = _ArchParams.from_hf_config(hf_config, config.tp)
        return cls(config, arch)
