"""Tests for the CPU fork override in :func:`vllm_simulated._use_fork_on_cpu`.

vLLM's CPU platform forces ``VLLM_WORKER_MULTIPROC_METHOD=spawn`` in
``CpuPlatform.check_and_update_config``, which makes every subprocess re-import
torch/vLLM and dominates cold-start time. The simulated model does no real
tensor compute, so it overrides that back to ``fork``. These tests verify the
override mechanism against the real ``CpuPlatform`` class.
"""

import inspect
import os

import pytest

pytest.importorskip("vllm")

import vllm_simulated
from vllm.platforms.cpu import CpuPlatform


def _install_fake_original(monkeypatch):
    """Replace check_and_update_config with a stand-in that forces spawn.

    This mimics the real CPU platform behavior (verified separately in
    ``test_real_cpu_platform_forces_spawn``) without constructing a full
    VllmConfig, so the wrapper logic can be exercised in isolation.
    """
    calls = []

    def fake_original(cls, vllm_config):
        calls.append(vllm_config)
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    monkeypatch.setattr(
        CpuPlatform, "check_and_update_config", classmethod(fake_original)
    )
    monkeypatch.delenv("VLLM_WORKER_MULTIPROC_METHOD", raising=False)
    return calls, fake_original


def test_real_cpu_platform_forces_spawn():
    """The premise: the real CPU platform sets the method to spawn."""
    src = inspect.getsource(CpuPlatform.check_and_update_config)
    assert "VLLM_WORKER_MULTIPROC_METHOD" in src
    assert "spawn" in src


def test_use_fork_on_cpu_overrides_spawn(monkeypatch):
    calls, _ = _install_fake_original(monkeypatch)
    monkeypatch.delenv("VLLM_SIM_MULTIPROC_METHOD", raising=False)

    vllm_simulated._use_fork_on_cpu()

    sentinel = object()
    CpuPlatform.check_and_update_config(sentinel)

    # The original ran (config still processed) but the method ends up fork.
    assert calls == [sentinel]
    assert os.environ["VLLM_WORKER_MULTIPROC_METHOD"] == "fork"


def test_opt_out_leaves_spawn(monkeypatch):
    _, fake_original = _install_fake_original(monkeypatch)
    monkeypatch.setenv("VLLM_SIM_MULTIPROC_METHOD", "spawn")

    vllm_simulated._use_fork_on_cpu()

    # No wrapping happened: the stand-in original is still in place.
    assert CpuPlatform.check_and_update_config.__func__ is fake_original
    CpuPlatform.check_and_update_config(object())
    assert os.environ["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"


def test_override_is_idempotent(monkeypatch):
    _install_fake_original(monkeypatch)
    monkeypatch.delenv("VLLM_SIM_MULTIPROC_METHOD", raising=False)

    vllm_simulated._use_fork_on_cpu()
    first = CpuPlatform.check_and_update_config.__func__
    vllm_simulated._use_fork_on_cpu()
    second = CpuPlatform.check_and_update_config.__func__

    # Second call must not wrap again.
    assert first is second
    CpuPlatform.check_and_update_config(object())
    assert os.environ["VLLM_WORKER_MULTIPROC_METHOD"] == "fork"
