"""Tests for simulated NIXL KV connector."""

import time
from unittest.mock import MagicMock, Mock

import pytest

pytest.importorskip("vllm")

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from transformers import AutoConfig

from vllm_simulated.kv_transfer_latency import KVTransferLatencyModel
from vllm_simulated.nixl_sim import (
    SimulatedNixlConnector,
    SimulatedNixlConnectorMetadata,
)


class TestKVTransferLatencyModel:
    """Test the KV transfer latency model."""

    def test_zero_tokens_returns_zero(self):
        """Zero tokens should have zero transfer time."""
        model = KVTransferLatencyModel(
            kv_bytes_per_token=1000.0,
            bandwidth_gbps=100.0,
            handshake_ms=2.0,
        )
        assert model.transfer_time_ms(0) == 0.0

    def test_handshake_is_floor(self):
        """Handshake time is the minimum transfer time."""
        model = KVTransferLatencyModel(
            kv_bytes_per_token=0.0,  # No data bytes
            bandwidth_gbps=100.0,
            handshake_ms=2.0,
        )
        assert model.transfer_time_ms(100) == 2.0

    def test_transfer_scales_with_tokens(self):
        """Transfer time should scale linearly with token count."""
        model = KVTransferLatencyModel(
            kv_bytes_per_token=1000.0,
            bandwidth_gbps=100.0,
            handshake_ms=0.0,
        )
        time_1 = model.transfer_time_ms(1)
        time_2 = model.transfer_time_ms(2)
        assert abs(time_2 - 2 * time_1) < 0.001

    def test_transfer_inverse_with_bandwidth(self):
        """Transfer time should be inversely proportional to bandwidth."""
        model_100 = KVTransferLatencyModel(
            kv_bytes_per_token=1000.0,
            bandwidth_gbps=100.0,
            handshake_ms=0.0,
        )
        model_200 = KVTransferLatencyModel(
            kv_bytes_per_token=1000.0,
            bandwidth_gbps=200.0,
            handshake_ms=0.0,
        )
        time_100 = model_100.transfer_time_ms(100)
        time_200 = model_200.transfer_time_ms(100)
        assert abs(time_100 - 2 * time_200) < 0.001

    def test_from_hf_config(self):
        """Test building latency model from HF config."""
        # Load a real config
        hf_config = AutoConfig.from_pretrained("tests/fixtures/sim-tiny")
        model = KVTransferLatencyModel.from_hf_config_and_extra(
            hf_config, tp=1, bandwidth_gbps=100.0, handshake_ms=2.0
        )

        # Should compute kv_bytes_per_token = L * 2 * dKV_full * 2 / tp
        # For sim-tiny: L=2, num_kv_heads=4, head_dim=32, so dKV_full=128
        # kv_bytes_per_token = 2 * 2 * 128 * 2 / 1 = 1024
        assert model.kv_bytes_per_token == 1024.0
        assert model.bandwidth_gbps == 100.0
        assert model.handshake_ms == 2.0


class TestSimulatedNixlConnector:
    """Test the simulated NIXL connector."""

    @pytest.fixture
    def mock_producer_config(self):
        """Create a mock VllmConfig for producer role."""
        config = MagicMock()
        config.model_config.hf_config = AutoConfig.from_pretrained(
            "tests/fixtures/sim-tiny"
        )
        config.parallel_config.tensor_parallel_size = 1
        kv_transfer_config = MagicMock()
        kv_transfer_config.engine_id = "test-engine"
        kv_transfer_config.is_kv_producer = True
        kv_transfer_config.is_kv_consumer = False
        kv_transfer_config.get_from_extra_config = Mock(
            side_effect=lambda k, d: {"bandwidth_gbps": 100.0, "handshake_ms": 2.0}.get(k, d)
        )
        config.kv_transfer_config = kv_transfer_config
        return config

    @pytest.fixture
    def mock_consumer_config(self):
        """Create a mock VllmConfig for consumer role."""
        config = MagicMock()
        config.model_config.hf_config = AutoConfig.from_pretrained(
            "tests/fixtures/sim-tiny"
        )
        config.parallel_config.tensor_parallel_size = 1
        kv_transfer_config = MagicMock()
        kv_transfer_config.engine_id = "test-engine"
        kv_transfer_config.is_kv_producer = False
        kv_transfer_config.is_kv_consumer = True
        kv_transfer_config.get_from_extra_config = Mock(
            side_effect=lambda k, d: {"bandwidth_gbps": 100.0, "handshake_ms": 2.0}.get(k, d)
        )
        config.kv_transfer_config = kv_transfer_config
        return config

    @pytest.fixture
    def mock_kv_cache_config(self):
        """Create a mock KVCacheConfig."""
        return MagicMock()

    def test_producer_request_finished_returns_params(
        self, mock_producer_config, mock_kv_cache_config
    ):
        """Producer request_finished should return delay_free_blocks and params dict."""
        connector = SimulatedNixlConnector(
            mock_producer_config,
            KVConnectorRole.SCHEDULER,
            mock_kv_cache_config,
        )

        request = MagicMock()
        request.request_id = "test-req"
        request.num_computed_tokens = 100
        request.kv_transfer_params = {}
        block_ids = [1, 2, 3]

        delay, params = connector.request_finished(request, block_ids)

        assert delay is True
        assert params is not None
        assert params["do_remote_decode"] is True
        assert params["do_remote_prefill"] is False
        assert params["remote_block_ids"] == block_ids
        assert params["remote_engine_id"] == "test-engine"
        assert params["remote_request_id"] == "test-req"
        assert params["remote_num_tokens"] == 100
        assert params["transfer_mode"] == "pull"
        assert "remote_host" in params
        assert "remote_port" in params
        assert "tp_size" in params

    def test_producer_request_finished_empty_blocks(
        self, mock_producer_config, mock_kv_cache_config
    ):
        """Producer with empty blocks should not delay freeing."""
        connector = SimulatedNixlConnector(
            mock_producer_config,
            KVConnectorRole.SCHEDULER,
            mock_kv_cache_config,
        )

        request = MagicMock()
        request.kv_transfer_params = {}
        block_ids = []

        delay, params = connector.request_finished(request, block_ids)

        assert delay is False
        assert params is None

    def test_consumer_get_num_new_matched_tokens_with_remote_prefill(
        self, mock_consumer_config, mock_kv_cache_config
    ):
        """Consumer should return tokens to load when do_remote_prefill is set."""
        connector = SimulatedNixlConnector(
            mock_consumer_config,
            KVConnectorRole.SCHEDULER,
            mock_kv_cache_config,
        )

        request = MagicMock()
        request.num_prompt_tokens = 100
        request.kv_transfer_params = {"do_remote_prefill": True}

        count, is_async = connector.get_num_new_matched_tokens(request, 10)

        assert count == 90
        assert is_async is True

    def test_consumer_get_num_new_matched_tokens_without_remote_prefill(
        self, mock_consumer_config, mock_kv_cache_config
    ):
        """Consumer should return 0 when do_remote_prefill is not set."""
        connector = SimulatedNixlConnector(
            mock_consumer_config,
            KVConnectorRole.SCHEDULER,
            mock_kv_cache_config,
        )

        request = MagicMock()
        request.kv_transfer_params = {}

        count, is_async = connector.get_num_new_matched_tokens(request, 10)

        assert count == 0
        assert is_async is False

    def test_consumer_update_state_after_alloc(
        self, mock_consumer_config, mock_kv_cache_config
    ):
        """Consumer should track requests to receive."""
        connector = SimulatedNixlConnector(
            mock_consumer_config,
            KVConnectorRole.SCHEDULER,
            mock_kv_cache_config,
        )

        request = MagicMock()
        request.request_id = "test-req"
        request.kv_transfer_params = {"do_remote_prefill": True}
        blocks = MagicMock()

        connector.update_state_after_alloc(request, blocks, 50)

        # Should clear do_remote_prefill
        assert request.kv_transfer_params["do_remote_prefill"] is False
        # Should have pending recv
        assert "test-req" in connector._pending_recv

    def test_worker_get_finished_reports_after_deadline(
        self, mock_consumer_config, mock_kv_cache_config
    ):
        """Worker get_finished should report requests after their deadline."""
        connector = SimulatedNixlConnector(
            mock_consumer_config,
            KVConnectorRole.WORKER,
            mock_kv_cache_config,
        )

        # Simulate a load deadline in the past
        connector._load_deadlines["req-1"] = time.monotonic() - 1.0

        done_sending, done_recving = connector.get_finished(set())

        assert done_recving == {"req-1"}
        # Should be cleaned up
        assert "req-1" not in connector._load_deadlines

    def test_worker_get_finished_does_not_report_before_deadline(
        self, mock_consumer_config, mock_kv_cache_config
    ):
        """Worker get_finished should not report requests before deadline."""
        connector = SimulatedNixlConnector(
            mock_consumer_config,
            KVConnectorRole.WORKER,
            mock_kv_cache_config,
        )

        # Simulate a load deadline in the future
        connector._load_deadlines["req-1"] = time.monotonic() + 10.0

        done_sending, done_recving = connector.get_finished(set())

        # Should not be reported yet
        assert done_recving == set() or done_recving is None
        # Should still be tracked
        assert "req-1" in connector._load_deadlines

    def test_worker_start_load_kv_sets_deadlines(
        self, mock_consumer_config, mock_kv_cache_config
    ):
        """Worker start_load_kv should set deadlines for requests."""
        connector = SimulatedNixlConnector(
            mock_consumer_config,
            KVConnectorRole.WORKER,
            mock_kv_cache_config,
        )

        # Create metadata with requests to load
        metadata = SimulatedNixlConnectorMetadata([("req-1", 100), ("req-2", 200)])
        connector.bind_connector_metadata(metadata)

        forward_context = MagicMock()
        connector.start_load_kv(forward_context)

        # Should have deadlines set
        assert "req-1" in connector._load_deadlines
        assert "req-2" in connector._load_deadlines
        # req-2 should have a later deadline (more tokens)
        assert connector._load_deadlines["req-2"] > connector._load_deadlines["req-1"]

    def test_build_connector_meta_drains_pending_recv(
        self, mock_consumer_config, mock_kv_cache_config
    ):
        """build_connector_meta should drain pending_recv into metadata."""
        connector = SimulatedNixlConnector(
            mock_consumer_config,
            KVConnectorRole.SCHEDULER,
            mock_kv_cache_config,
        )

        # Add some pending recvs
        req1 = MagicMock()
        req1.request_id = "req-1"
        req2 = MagicMock()
        req2.request_id = "req-2"
        connector._pending_recv["req-1"] = (req1, 100)
        connector._pending_recv["req-2"] = (req2, 200)

        scheduler_output = MagicMock()
        metadata = connector.build_connector_meta(scheduler_output)

        assert isinstance(metadata, SimulatedNixlConnectorMetadata)
        assert len(metadata.reqs_to_load) == 2
        assert ("req-1", 100) in metadata.reqs_to_load
        assert ("req-2", 200) in metadata.reqs_to_load
        # Should be cleared
        assert len(connector._pending_recv) == 0
