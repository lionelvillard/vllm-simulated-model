"""Latency model for simulated KV transfer."""

from dataclasses import dataclass

from vllm_simulated.physics_latency import _ArchParams, _ABPP


@dataclass(frozen=True)
class KVTransferLatencyModel:
    """Bandwidth-based KV transfer latency model.

    Models transfer time as: handshake_ms + (kv_bytes / bandwidth).
    KV bytes are computed from model architecture and number of tokens.
    """

    kv_bytes_per_token: float
    bandwidth_gbps: float
    handshake_ms: float

    def transfer_time_ms(self, num_tokens: int) -> float:
        """Compute transfer time in milliseconds for given number of tokens.

        Args:
            num_tokens: Number of tokens to transfer.

        Returns:
            Transfer time in milliseconds.
        """
        if num_tokens <= 0:
            return 0.0

        kv_bytes = num_tokens * self.kv_bytes_per_token
        # bandwidth_gbps * 1e9 converts to bytes/sec, / 1000 gives bytes/ms
        transfer_ms = kv_bytes / (self.bandwidth_gbps * 1e9 / 1000.0)
        return self.handshake_ms + transfer_ms

    @classmethod
    def from_hf_config_and_extra(
        cls,
        hf_config,
        tp: int,
        bandwidth_gbps: float = 100.0,
        handshake_ms: float = 2.0,
    ) -> "KVTransferLatencyModel":
        """Build latency model from HF config and extra parameters.

        Args:
            hf_config: HuggingFace model config with architecture parameters.
            tp: Tensor parallelism degree.
            bandwidth_gbps: Network bandwidth in Gbps. Default 100.
            handshake_ms: Handshake overhead in milliseconds. Default 2.0.

        Returns:
            KVTransferLatencyModel instance.
        """
        arch = _ArchParams.from_hf_config(hf_config, tp)
        # KV bytes per token = L * 2 (K+V) * dKV_full * kv_dtype_bytes / tp
        kv_bytes_per_token = arch.L * 2 * arch.dKV_full * _ABPP / tp
        return cls(
            kv_bytes_per_token=kv_bytes_per_token,
            bandwidth_gbps=bandwidth_gbps,
            handshake_ms=handshake_ms,
        )

    @classmethod
    def from_vllm_config(cls, vllm_config) -> "KVTransferLatencyModel":
        """Build latency model from vLLM config.

        Args:
            vllm_config: VllmConfig with model_config, parallel_config, and
                kv_transfer_config.

        Returns:
            KVTransferLatencyModel instance.
        """
        hf_config = vllm_config.model_config.hf_config
        tp = vllm_config.parallel_config.tensor_parallel_size
        kv_transfer_config = vllm_config.kv_transfer_config

        bandwidth_gbps = kv_transfer_config.get_from_extra_config(
            "bandwidth_gbps", 100.0
        )
        handshake_ms = kv_transfer_config.get_from_extra_config(
            "handshake_ms", 2.0
        )

        return cls.from_hf_config_and_extra(
            hf_config, tp, bandwidth_gbps, handshake_ms
        )
