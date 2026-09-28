"""Teacher prompt-score parsing uses input-token identities, not dict order."""

import asyncio

import httpx
import pytest
from openai import AsyncOpenAI

from prime_rl.orchestrator.clients import prefill_logprobs


def score(entries):
    async def run():
        transport = httpx.MockTransport(
            lambda request: httpx.Response(200, json={"choices": [], "prompt_logprobs": entries})
        )
        async with AsyncOpenAI(
            base_url="http://teacher/v1",
            api_key="EMPTY",
            http_client=httpx.AsyncClient(transport=transport),
        ) as client:
            return await prefill_logprobs(client, "teacher", [10, 20])

    return asyncio.run(run())


def test_teacher_score_uses_actual_token_when_top_token_comes_first():
    assert score([None, {"30": {"logprob": -0.1}, "20": {"logprob": -3.0}}]) == [0.0, -3.0]


@pytest.mark.parametrize("entries", [None, [None], [None, {}], [None, {"30": {"logprob": -0.1}}]])
def test_teacher_score_rejects_missing_targets(entries):
    with pytest.raises(ValueError):
        score(entries)
