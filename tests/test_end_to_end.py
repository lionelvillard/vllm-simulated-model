import time as _time

import pytest

pytest.importorskip("vllm")

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

import vllm_simulated


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


def test_decode_latency_lower_bound(sim_llm):
    # sim-tiny: base_ms=1, decode_ms_per_seq=2 -> each decode step sleeps ~3ms
    # for a single sequence. With max_tokens=10 there are ~10 steps.
    start = _time.perf_counter()
    sim_llm.generate(
        TokensPrompt(prompt_token_ids=[1, 2, 3, 4]),
        SamplingParams(max_tokens=10),
    )
    elapsed_ms = (_time.perf_counter() - start) * 1000.0
    # time.sleep guarantees AT LEAST the requested duration, so a lower bound
    # is non-flaky. 10 decode steps * 3ms = 30ms; allow slack for the prefill
    # step and scheduling. Assert we spent at least half the modeled decode time.
    assert elapsed_ms >= 10 * 3.0 * 0.5
