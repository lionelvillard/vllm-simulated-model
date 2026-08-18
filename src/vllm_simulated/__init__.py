# register() is added in Task 2.


def register() -> None:
    """Entry point for the `vllm.general_plugins` group."""
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "SimulatedForCausalLM",
        "vllm_simulated.model:SimulatedForCausalLM",
    )
