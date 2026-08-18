from dataclasses import dataclass, fields


@dataclass(frozen=True)
class LatencyConfig:
    base_ms: float = 0.0
    prefill_ms_per_token: float = 0.0
    decode_ms_per_seq: float = 0.0
    ctx_ms_per_ktoken: float = 0.0
    deterministic_length: bool = True

    _COEFFICIENTS = (
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
