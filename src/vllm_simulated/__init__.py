import os


def register() -> None:
    """Entry point for the `vllm.general_plugins` group."""
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "SimulatedForCausalLM",
        "vllm_simulated.model:SimulatedForCausalLM",
    )
    _use_fork_on_cpu()


def _use_fork_on_cpu() -> None:
    """Use ``fork`` instead of ``spawn`` for CPU worker/engine subprocesses.

    vLLM's CPU platform unconditionally sets
    ``VLLM_WORKER_MULTIPROC_METHOD=spawn`` in
    ``CpuPlatform.check_and_update_config`` because forking a process that has
    already initialized OpenMP is unsafe for real tensor compute. ``spawn``
    forces every subprocess (the engine core and each worker) to re-import
    torch and vLLM from scratch, which dominates cold-start time (~40s on the
    CPU image).

    The simulated model performs no real tensor compute, so ``fork`` is safe
    here: children inherit the parent's already-imported modules via
    copy-on-write instead of re-importing them.

    ``register`` runs early in every process (API server, engine core, and
    worker), before ``check_and_update_config``, so wrapping that method
    ensures the ``fork`` setting is re-applied after vLLM's CPU platform forces
    ``spawn``.

    Set ``VLLM_SIM_MULTIPROC_METHOD=spawn`` to restore vLLM's default behavior.
    """
    method = os.environ.get("VLLM_SIM_MULTIPROC_METHOD", "fork")
    if method != "fork":
        # Only "fork" is worth overriding; anything else means "leave vLLM's
        # CPU default (spawn) alone".
        return

    try:
        from vllm.platforms.cpu import CpuPlatform
    except Exception:
        # Not a CPU build / platform unavailable: nothing to override.
        return

    original = CpuPlatform.check_and_update_config.__func__
    if getattr(original, "_sim_fork_patched", False):
        return

    def check_and_update_config(cls, vllm_config) -> None:
        original(cls, vllm_config)
        # vLLM's CPU platform just forced "spawn"; override it back to "fork".
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = method

    check_and_update_config._sim_fork_patched = True
    CpuPlatform.check_and_update_config = classmethod(check_and_update_config)
