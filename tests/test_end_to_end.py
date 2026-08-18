import pytest

pytest.importorskip("vllm")

from vllm import LLM, SamplingParams  # noqa: E402
from vllm.inputs import TokensPrompt  # noqa: E402

import vllm_simulated  # noqa: E402


@pytest.fixture(scope="module")
def sim_llm(tmp_path_factory, monkeypatch_module):
    monkeypatch_module.setenv("VLLM_CPU_KVCACHE_SPACE", "1")
    vllm_simulated.register()
    return LLM(
        model="tests/fixtures/sim-tiny",
        load_format="dummy",
        skip_tokenizer_init=True,
        enforce_eager=True,
    )


def test_output_length_honors_max_tokens(sim_llm):
    outputs = sim_llm.generate(
        TokensPrompt(prompt_token_ids=[1, 2, 3, 4]),
        SamplingParams(max_tokens=8),
    )
    # deterministic_length masks EOS, so the request runs to max_tokens.
    assert len(outputs[0].outputs[0].token_ids) == 8
