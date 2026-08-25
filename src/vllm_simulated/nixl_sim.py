"""Simulated NIXL KV connector for vLLM prefill/decode disaggregation.

This connector replicates the HTTP kv_transfer_params contract used by the
llm-d sidecar, but skips the actual NIXL/UCX RDMA transfer. Instead, it uses
a bandwidth-based latency model to simulate transfer time via time.monotonic()
deadlines.
"""

import os
import time
from typing import TYPE_CHECKING, Any

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.logger import init_logger

from vllm_simulated.kv_transfer_latency import KVTransferLatencyModel

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


class SimulatedNixlConnectorMetadata(KVConnectorMetadata):
    """Metadata for simulated NIXL connector.

    Carries the list of requests and their token counts that need to be
    fake-loaded on the consumer side.
    """

    def __init__(self, reqs_to_load: list[tuple[str, int]]):
        """Initialize metadata.

        Args:
            reqs_to_load: List of (request_id, num_tokens) tuples to load.
        """
        self.reqs_to_load = reqs_to_load


class SimulatedNixlConnector(KVConnectorBase_V1):
    """Simulated NIXL connector for P/D disaggregation.

    Handles both producer (prefill) and consumer (decode) roles. Uses
    time.monotonic() deadlines to simulate KV transfer time based on a
    bandwidth model, without actually transferring data.
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ):
        """Initialize the simulated NIXL connector.

        Args:
            vllm_config: vLLM configuration.
            role: SCHEDULER or WORKER role.
            kv_cache_config: KV cache configuration.
        """
        super().__init__(vllm_config, role, kv_cache_config)

        self._latency_model = KVTransferLatencyModel.from_vllm_config(vllm_config)

        # Scheduler-side state
        self._pending_recv: dict[str, tuple["Request", int]] = {}  # req_id -> (req, num_tokens)
        self._pending_send: dict[str, float] = {}  # req_id -> deadline (monotonic time)

        # Worker-side state
        self._load_deadlines: dict[str, float] = {}  # req_id -> deadline (monotonic time)
        self._send_deadlines: dict[str, float] = {}  # req_id -> deadline (monotonic time)

        # Side-channel config (kept for wire compatibility)
        self._side_channel_host = os.environ.get(
            "VLLM_NIXL_SIDE_CHANNEL_HOST", "localhost"
        )
        self._side_channel_port = int(
            os.environ.get("VLLM_NIXL_SIDE_CHANNEL_PORT", "5600")
        )

        logger.info(
            "SimulatedNixlConnector initialized: role=%s, bandwidth=%.1f Gbps, "
            "handshake=%.1f ms, kv_bytes_per_token=%.1f",
            role,
            self._latency_model.bandwidth_gbps,
            self._latency_model.handshake_ms,
            self._latency_model.kv_bytes_per_token,
        )

    # ==============================
    # Worker-side methods
    # ==============================

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        """Start loading KV cache (simulated via deadlines).

        For each request in the connector metadata, computes a transfer time
        and stores a deadline. No actual transfer happens.

        Args:
            forward_context: Forward context.
            **kwargs: Additional arguments (unused).
        """
        if not self.has_connector_metadata():
            return

        metadata = self._get_connector_metadata()
        if not isinstance(metadata, SimulatedNixlConnectorMetadata):
            return

        now = time.monotonic()
        for req_id, num_tokens in metadata.reqs_to_load:
            transfer_ms = self._latency_model.transfer_time_ms(num_tokens)
            deadline = now + transfer_ms / 1000.0
            self._load_deadlines[req_id] = deadline
            logger.debug(
                "SimulatedNixlConnector: scheduling load for req=%s, tokens=%d, "
                "transfer_ms=%.2f, deadline=%.3f",
                req_id,
                num_tokens,
                transfer_ms,
                deadline,
            )

    def wait_for_layer_load(self, layer_name: str) -> None:
        """Wait for layer load (no-op for simulated connector).

        Args:
            layer_name: Layer name.
        """
        pass

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: "torch.Tensor",  # noqa: F821
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        """Save KV layer (no-op for simulated connector).

        Args:
            layer_name: Layer name.
            kv_layer: KV cache tensor.
            attn_metadata: Attention metadata.
            **kwargs: Additional arguments.
        """
        pass

    def wait_for_save(self):
        """Wait for save operations (no-op for simulated connector)."""
        pass

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        """Return requests that have finished async transfer.

        Checks deadlines and returns requests whose transfer window has passed.

        Args:
            finished_req_ids: Request IDs that finished generating tokens.

        Returns:
            (done_sending, done_recving) tuple of request ID sets.
        """
        now = time.monotonic()
        done_sending = None
        done_recving = None

        # Check consumer (load) deadlines
        if self._load_deadlines:
            done_recving = {
                req_id
                for req_id, deadline in self._load_deadlines.items()
                if now >= deadline
            }
            # Clean up finished requests
            for req_id in done_recving:
                del self._load_deadlines[req_id]
                logger.debug(
                    "SimulatedNixlConnector: finished loading req=%s", req_id
                )

        # Check producer (send) deadlines
        if self._send_deadlines:
            done_sending = {
                req_id
                for req_id, deadline in self._send_deadlines.items()
                if now >= deadline
            }
            # Clean up finished requests
            for req_id in done_sending:
                del self._send_deadlines[req_id]
                logger.debug(
                    "SimulatedNixlConnector: finished sending req=%s", req_id
                )

        return done_sending, done_recving

    # ==============================
    # Scheduler-side methods
    # ==============================

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        """Get number of tokens to load from external KV cache.

        For consumer (decode): if do_remote_prefill is set, returns the number
        of prompt tokens beyond what's already computed.
        For producer (prefill): always returns (0, False).

        Args:
            request: Request object.
            num_computed_tokens: Number of locally computed tokens.

        Returns:
            (num_tokens_to_load, is_async) tuple.
        """
        params = request.kv_transfer_params
        if params is None:
            return 0, False

        # Consumer (decode) side: check if remote prefill is requested
        if params.get("do_remote_prefill"):
            num_prompt_tokens = request.num_prompt_tokens
            count = num_prompt_tokens - num_computed_tokens
            if count > 0:
                logger.debug(
                    "SimulatedNixlConnector: remote prefill for req=%s, "
                    "num_prompt_tokens=%d, num_computed_tokens=%d, count=%d",
                    request.request_id,
                    num_prompt_tokens,
                    num_computed_tokens,
                    count,
                )
                return count, True

        # Producer (prefill) side or no remote prefill
        return 0, False

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        """Update connector state after block allocation.

        For consumer (decode): records requests that need to receive KV from
        the producer and marks do_remote_prefill as False to prevent re-transfer.

        Args:
            request: Request object.
            blocks: Allocated KV cache blocks.
            num_external_tokens: Number of tokens to load from external cache.
        """
        params = request.kv_transfer_params
        if params is None:
            return

        # Consumer side: track requests to receive
        if params.get("do_remote_prefill") and num_external_tokens > 0:
            self._pending_recv[request.request_id] = (request, num_external_tokens)
            # Set do_remote_prefill to False to prevent re-transfer
            params["do_remote_prefill"] = False
            logger.debug(
                "SimulatedNixlConnector: scheduled recv for req=%s, tokens=%d",
                request.request_id,
                num_external_tokens,
            )

    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> KVConnectorMetadata:
        """Build connector metadata for this step.

        Drains the pending_recv map and packages it into metadata for the
        worker to process.

        Args:
            scheduler_output: Scheduler output.

        Returns:
            SimulatedNixlConnectorMetadata with requests to load.
        """
        reqs_to_load = []
        for req_id, (req, num_tokens) in self._pending_recv.items():
            reqs_to_load.append((req_id, num_tokens))

        # Clear pending recv after packaging
        self._pending_recv.clear()

        return SimulatedNixlConnectorMetadata(reqs_to_load)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """Called when a request has finished.

        For producer (prefill): returns kv_transfer_params dict and delays
        block freeing by setting a deadline. The blocks are held until the
        transfer window expires.
        For consumer (decode): returns False, None (no transfer).

        Args:
            request: Request object.
            block_ids: KV cache block IDs for this request.

        Returns:
            (delay_free_blocks, kv_transfer_params) tuple.
        """
        params = request.kv_transfer_params
        if params is None:
            return False, None

        is_producer = self._kv_transfer_config.is_kv_producer

        if not is_producer:
            # Consumer (decode) side: no transfer on finish
            return False, None

        # Producer (prefill) side: prepare kv_transfer_params
        delay_free_blocks = len(block_ids) > 0
        if not delay_free_blocks:
            return False, None

        # Calculate transfer time and set deadline
        num_tokens = request.num_computed_tokens
        transfer_ms = self._latency_model.transfer_time_ms(num_tokens)
        deadline = time.monotonic() + transfer_ms / 1000.0
        self._pending_send[request.request_id] = deadline

        # Also set worker-side deadline for get_finished
        self._send_deadlines[request.request_id] = deadline

        logger.debug(
            "SimulatedNixlConnector: request_finished req=%s, tokens=%d, "
            "transfer_ms=%.2f, deadline=%.3f, blocks=%d",
            request.request_id,
            num_tokens,
            transfer_ms,
            deadline,
            len(block_ids),
        )

        # Build kv_transfer_params dict (mirrors pull_scheduler.py:269-281)
        return delay_free_blocks, dict(
            do_remote_prefill=False,
            do_remote_decode=True,
            remote_block_ids=block_ids,
            remote_engine_id=self._kv_transfer_config.engine_id,
            remote_request_id=request.request_id,
            remote_host=self._side_channel_host,
            remote_port=self._side_channel_port,
            tp_size=self._vllm_config.parallel_config.tensor_parallel_size,
            remote_num_tokens=num_tokens,
            remote_blocks_expiry_time=deadline,
            transfer_mode="pull",
        )
