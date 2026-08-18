from dataclasses import dataclass, fields
from typing import ClassVar, Protocol


class LatencyModel(Protocol):
    def step_time_ms(self, shape: "BatchShape") -> float: ...


@dataclass(frozen=True)
class LatencyConfig:
    base_ms: float = 0.0
    prefill_ms_per_token: float = 0.0
    decode_ms_per_seq: float = 0.0
    ctx_ms_per_ktoken: float = 0.0
    deterministic_length: bool = True

    _COEFFICIENTS: ClassVar[tuple[str, ...]] = (
        "base_ms",
        "prefill_ms_per_token",
        "decode_ms_per_seq",
        "ctx_ms_per_ktoken",
    )

    @classmethod
    def from_dict(cls, d: dict) -> "LatencyConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(
                f"Unknown latency config keys: {sorted(unknown)}"
            )
        config = cls(**d)
        config.validate()
        return config

    def validate(self) -> None:
        for name in self._COEFFICIENTS:
            value = getattr(self, name)
            if value < 0:
                raise ValueError(
                    f"latency.{name} must be >= 0, got {value}"
                )


@dataclass(frozen=True)
class BatchShape:
    num_prefill_tokens: int
    num_decode_seqs: int
    sum_context_len: int


class SimulatedLatencyModel:
    def __init__(self, config: LatencyConfig) -> None:
        self.config = config

    def step_time_ms(self, shape: BatchShape) -> float:
        c = self.config
        total = (
            c.base_ms
            + c.prefill_ms_per_token * shape.num_prefill_tokens
            + c.decode_ms_per_seq * shape.num_decode_seqs
            + c.ctx_ms_per_ktoken * (shape.sum_context_len / 1000.0)
        )
        return max(0.0, total)

    @classmethod
    def from_dict(cls, d: dict, **kwargs) -> "SimulatedLatencyModel":
        return cls(LatencyConfig.from_dict(d))


_REGISTRY: dict[str, type] = {
    "linear": SimulatedLatencyModel,
}


def build_latency_model(d: dict, hf_config=None) -> LatencyModel:
    d = dict(d)  # don't mutate caller's dict
    model_type = d.pop("type", "linear")
    cls = _REGISTRY.get(model_type)
    if cls is None:
        raise ValueError(
            f"Unknown latency model type: {model_type!r}. "
            f"Known types: {sorted(_REGISTRY)}"
        )
    return cls.from_dict(d, hf_config=hf_config)


def batch_shape_from_attn_metadata(md) -> BatchShape:
    query_start_loc = md.query_start_loc
    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    is_decode = query_lens <= 1
    num_decode_seqs = int(is_decode.sum().item())
    num_prefill_tokens = int(query_lens[~is_decode].sum().item())
    sum_context_len = int(md.seq_lens.sum().item())
    return BatchShape(
        num_prefill_tokens=num_prefill_tokens,
        num_decode_seqs=num_decode_seqs,
        sum_context_len=sum_context_len,
    )
