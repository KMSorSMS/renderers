"""Native SGLang wire-contract tests; no engine, GPU, or tokenizer downloads."""

import asyncio
import base64
import copy
import json

import httpx
import numpy as np
import pytest
from openai import AsyncOpenAI
from pydantic import ValidationError

from renderers.backends import EngineAbortedError, EngineAbortError, SGLangOptions
from renderers.base import (
    MultiModalData,
    ParsedResponse,
    PlaceholderRange,
    RenderedTokens,
)
from renderers.client import (
    MalformedGenerateResponseError,
    OverlongPromptError,
    generate,
)


class Renderer:
    supports_tools = True

    def render(self, messages, **kwargs):
        return RenderedTokens(token_ids=[1, 2, 3])

    def get_stop_token_ids(self):
        return [99]

    def parse_response(self, ids, *, tools=None, prompt_ids=None):
        assert prompt_ids == [1, 2, 3]
        return ParsedResponse(content="decoded")


def payload():
    return {
        "output_ids": [7, 8],
        "meta_info": {
            "id": "engine-id",
            "finish_reason": {"type": "stop", "matched": 99},
            "prompt_tokens": 3,
            "completion_tokens": 2,
            "output_token_logprobs": [[-0.1, 7, None], [-0.2, 8, None]],
        },
    }


async def request(response=None, **kwargs):
    captured = []

    async def handler(req):
        captured.append(req)
        return httpx.Response(200, json=response if response is not None else payload())

    async with AsyncOpenAI(
        base_url="http://engine/v1",
        api_key="test",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    ) as client:
        result = await generate(
            client=client,
            renderer=Renderer(),
            model="test",
            messages=[{"role": "user", "content": "hello"}],
            backend="sglang",
            **kwargs,
        )
    return result, captured


@pytest.mark.asyncio
@pytest.mark.parametrize("options", [{}, {"max_prompt_len": 4, "max_context_len": 8}])
async def test_sglang_does_not_discover_vllm_model_limits(options):
    result, requests = await request(backend_options=options)
    assert result["completion_ids"] == [7, 8]
    assert [(req.method, req.url.path) for req in requests] == [("POST", "/generate")]


@pytest.mark.asyncio
async def test_explicit_common_prompt_limit_still_applies_to_sglang():
    with pytest.raises(OverlongPromptError):
        await request(max_prompt_len=2, is_initial_turn=False)


@pytest.mark.asyncio
async def test_native_request_is_token_exact_and_preserves_options():
    sampling = {
        "max_tokens": 8,
        "min_tokens": 1,
        "seed": 42,
        "temperature": 0.7,
        "stop_token_ids": [101],
        "stop": ["user-stop"],
        "top_p": 0.95,
    }
    original = copy.deepcopy(sampling)
    result, requests = await request(
        sampling_params=sampling,
        cache_salt="run-1",
        priority=3,
        extra_headers={"x-session-id": "session", "x-custom": "a"},
        backend_options={
            "lora_path": "adapter",
            "routing_header": "x-routing-key",
            "max_response_tokens": 5,
            "max_context_len": 7,
            "stop_token_ids": [102, 99],
            "stop": ["launch-stop"],
            "skip_special_tokens": False,
        },
    )
    assert sampling == original
    assert len(requests) == 1 and requests[0].url.path == "/generate"
    body = json.loads(requests[0].content)
    assert body["input_ids"] == [1, 2, 3]
    assert body["sampling_params"] == {
        "max_new_tokens": 4,
        "min_new_tokens": 1,
        "sampling_seed": 42,
        "temperature": 0.7,
        "top_p": 0.95,
        "stop_token_ids": [99, 101, 102],
        "stop": ["user-stop", "launch-stop"],
        "skip_special_tokens": False,
        "spaces_between_special_tokens": False,
        "no_stop_trim": True,
        "n": 1,
    }
    assert body["lora_path"] == "adapter"
    assert body["extra_key"] == "run-1" and body["priority"] == 3
    assert body["session_id"] == body["routing_key"] == "session"
    assert requests[0].headers["x-routing-key"] == "session"
    assert requests[0].headers["x-custom"] == "a"
    assert body["return_logprob"] and not body["return_text_in_logprobs"]
    assert result["completion_ids"] == [7, 8]
    np.testing.assert_allclose(result["completion_logprobs"], [-0.1, -0.2])
    assert result["completion_logprobs"].dtype == np.float32
    assert result["request_id"] == "engine-id"
    assert result["finish_reason"] == "stop"
    assert result["usage"]["total_tokens"] == 5


@pytest.mark.asyncio
async def test_full_training_does_not_send_lora_path():
    _, requests = await request()
    assert "lora_path" not in json.loads(requests[0].content)


@pytest.mark.asyncio
@pytest.mark.parametrize("encoded", [False, True, None])
async def test_top_head_compact_and_native_formats_agree(encoded):
    data = payload()
    ids = np.asarray([[7, 10, 11, 12], [8, 10, 11, 12]], dtype="<i4")
    probs = np.asarray([[-0.1, -4, -5, -6], [-0.2, -4, -5, -6]], dtype="<f4")
    meta = data["meta_info"]
    if encoded is None:
        meta["output_top_logprobs"] = [
            [[float(p), int(i), None] for i, p in zip(irow, prow)]
            for irow, prow in zip(ids, probs)
        ]
    else:
        meta["output_top_logprobs_shape"] = [2, 4]
        meta["output_top_logprobs_null_prefix"] = 0
        for name, array, dtype in (("idx", ids, "int32"), ("val", probs, "float32")):
            key = f"output_top_logprobs_{name}_flat"
            if encoded:
                meta[key + "_b64"] = base64.b64encode(array.tobytes()).decode()
                meta[key + "_b64_dtype"] = dtype
            else:
                meta[key] = array.flatten().tolist()
    result, requests = await request(data, sampling_params={"logprobs": 2})
    assert result["completion_top_ids"] == [[7, 10], [8, 10]]
    np.testing.assert_allclose(
        result["completion_top_logprobs"], [[-0.1, -4], [-0.2, -4]]
    )
    body = json.loads(requests[0].content)
    assert body["top_logprobs_num"] == 4
    assert body["return_flat_raw_top_logprobs"]
    assert body["return_flat_raw_output_top_logprobs"]
    assert body["return_flat_raw_top_logprobs_b64"]
    assert "extra_args" not in body["sampling_params"]


@pytest.mark.asyncio
async def test_sampling_support_uses_behavior_probabilities():
    data = payload()
    data["meta_info"].update(
        output_token_sampling_mask=[[7, 10], [8, 11]],
        output_token_sampling_logprobs=[
            [np.log(0.8), np.log(0.2)],
            [np.log(0.7), np.log(0.3)],
        ],
    )
    result, _ = await request(data, backend_options={"return_sampling_mask": True})
    assert result["sampling_mask"] == [[7, 10], [8, 11]]
    np.testing.assert_allclose(result["completion_logprobs"], np.log([0.8, 0.7]))


@pytest.mark.asyncio
async def test_replay_buffers_and_weight_spans_keep_token_alignment():
    data = payload()
    array = np.arange(8, dtype="<i4").reshape(4, 1, 2)
    data["meta_info"].update(
        routed_experts=base64.b64encode(array.tobytes()).decode(),
        indexer_topk=base64.b64encode(array.tobytes()).decode(),
        weight_versions=[
            {"version": "v1", "start": 0, "end": 1},
            {"version": "v2", "start": 1, "end": 2},
        ],
    )
    options = {
        "return_routed_experts": True,
        "routed_experts_shape": [1, 2],
        "return_indexer_topk": True,
        "indexer_topk_shape": [1, 2],
    }
    result, _ = await request(data, backend_options=options)
    for key in ("routed_experts", "indexer_topk"):
        assert result[key]["shape"] == [4, 1, 2]
        assert result[key]["dtype"] == "int32" and result[key]["start"] == 0
        np.testing.assert_array_equal(
            np.frombuffer(base64.b64decode(result[key]["data"]), dtype="<i4").reshape(
                4, 1, 2
            ),
            array,
        )
    assert result["weight_versions"] == data["meta_info"]["weight_versions"]


@pytest.mark.asyncio
async def test_incremental_routing_starts_at_requested_token():
    data = payload()
    data["meta_info"]["routed_experts"] = base64.b64encode(
        np.asarray([1, 2], dtype="<i4").tobytes()
    ).decode()
    result, requests = await request(
        data,
        sampling_params={"routed_experts_prompt_start": 2},
        backend_options={"return_routed_experts": True, "routed_experts_shape": [1, 1]},
    )
    assert result["routed_experts"]["start"] == 2
    assert result["routed_experts"]["shape"] == [2, 1, 1]
    assert json.loads(requests[0].content)["routed_experts_start_len"] == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.update(output_ids=[8, 7]),
        lambda d: d.update(output_ids=[7.0, 8.0]),
        lambda d: d["meta_info"].update(
            output_token_logprobs=[[-1e39, 7, None], [-0.2, 8, None]]
        ),
        lambda d: d["meta_info"].update(completion_tokens=2.0),
        lambda d: d["meta_info"].update(completion_tokens=3),
        lambda d: d["meta_info"].update(output_token_logprobs=None),
        lambda d: d["meta_info"].update(
            output_token_logprobs=[[None, 7, None], [-0.2, 8, None]]
        ),
        lambda d: d["meta_info"].update(
            output_token_logprobs=[[0.1, 7, None], [-0.2, 8, None]]
        ),
        lambda d: d["meta_info"].update(
            output_token_logprobs=[[-0.1, True, None], [-0.2, 8, None]]
        ),
        lambda d: d["meta_info"].update(finish_reason={"type": "surprise"}),
        lambda d: d["meta_info"].update(prompt_tokens=4),
        lambda d: d["meta_info"].update(
            weight_versions=[{"version": "v1", "start": 1, "end": 2}]
        ),
    ],
)
async def test_malformed_evidence_fails_instead_of_training_on_corrupt_tokens(mutate):
    data = payload()
    mutate(data)
    with pytest.raises(MalformedGenerateResponseError):
        await request(data)


@pytest.mark.asyncio
async def test_explicit_abort_is_not_success():
    data = payload()
    data["meta_info"]["finish_reason"] = {"type": "abort", "message": "preempted"}
    with pytest.raises(EngineAbortedError):
        await request(data)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "sampling",
    [
        {"extra_args": {"prl_compact_logprobs": True}},
        {"n": 3},
        {"seed": 1, "sampling_seed": 2},
        {"no_stop_trim": False},
    ],
)
async def test_unsupported_sampling_is_not_silently_discarded(sampling):
    with pytest.raises(ValueError):
        await request(sampling_params=sampling)


@pytest.mark.asyncio
async def test_initial_prompt_limit_does_not_apply_to_later_turns():
    with pytest.raises(OverlongPromptError):
        await request(backend_options={"max_prompt_len": 2})
    result, _ = await request(
        backend_options={"max_prompt_len": 2}, is_initial_turn=False
    )
    assert result["completion_ids"] == [7, 8]
    with pytest.raises(OverlongPromptError):
        await request(backend_options={"max_context_len": 3}, is_initial_turn=False)


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", [False, True])
async def test_abandoned_generation_aborts_only_its_rid_with_same_routing(timeout):
    requests = []
    started = asyncio.Event()
    terminal = asyncio.Event()

    async def handler(req):
        requests.append(req)
        if req.url.path == "/generate":
            started.set()
            if timeout:
                raise httpx.ReadTimeout("timed out", request=req)
            await terminal.wait()
        elif req.url.path == "/abort_request":
            terminal.set()
        return httpx.Response(200, json={})

    async with AsyncOpenAI(
        base_url="http://engine/v1",
        api_key="test",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    ) as client:
        task = asyncio.create_task(
            generate(
                client=client,
                renderer=Renderer(),
                model="test",
                messages=[],
                backend="sglang",
                max_prompt_len=100,
                extra_headers={"x-session-id": "s"},
                backend_options={"routing_header": "route"},
            )
        )
        await started.wait()
        if not timeout:
            task.cancel()
        with pytest.raises(EngineAbortError if timeout else asyncio.CancelledError):
            await task
    assert len(requests) == 2
    first, abort = requests
    assert abort.url.path == "/abort_request"
    assert json.loads(abort.content) == {"rid": json.loads(first.content)["rid"]}
    assert abort.headers["route"] == first.headers["route"] == "s"


@pytest.mark.asyncio
@pytest.mark.parametrize("changed", [False, True])
async def test_multimodal_preserves_text_tokens_and_sidecar(changed):
    data = payload()
    data["prompt_token_ids"] = [10 if changed else 1, 999999, 3]
    captured = []
    sidecar = MultiModalData(
        mm_placeholders={"image": [PlaceholderRange(offset=1, length=1)]},
        mm_items={"image": [{"pixel_values": np.zeros((1, 3))}]},
    )

    async def handler(req):
        captured.append(req)
        return httpx.Response(200, json=data)

    async with AsyncOpenAI(
        base_url="http://engine/v1",
        api_key="test",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    ) as client:
        call = generate(
            client=client,
            renderer=Renderer(),
            model="test",
            backend="sglang",
            max_prompt_len=100,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.com/image.png"},
                        }
                    ],
                }
            ],
            prompt_ids=[1, 2, 3],
            multi_modal_data=sidecar,
        )
        if changed:
            with pytest.raises(MalformedGenerateResponseError):
                await call
        else:
            result = await call
            assert result["prompt_ids"] == [1, 2, 3]
            assert result["multi_modal_data"] is sidecar
            body = json.loads(captured[0].content)
            assert body["image_data"] == ["https://example.com/image.png"]
            assert body["return_prompt_token_ids"] is True
            assert "features" not in body


def test_options_reject_unknown_fields_and_unshaped_replay():
    for options in (
        {"unknown": True},
        {"return_routed_experts": True},
        {"routed_experts_shape": [0, 8]},
    ):
        with pytest.raises(ValidationError):
            SGLangOptions.model_validate(options)


@pytest.mark.asyncio
async def test_sglang_negative_logprobs_do_not_use_vllm_missing_sentinel():
    data = payload()
    data["meta_info"]["output_token_logprobs"][0][0] = -9999.0
    result, _ = await request(data)
    assert result["completion_logprobs"][0] == -9999.0


@pytest.mark.asyncio
async def test_compact_top_logprobs_reject_truncated_buffer():
    data = payload()
    data["meta_info"].update(
        output_top_logprobs_shape=[2, 4],
        output_top_logprobs_null_prefix=0,
        output_top_logprobs_idx_flat_b64="AA==",
        output_top_logprobs_idx_flat_b64_dtype="int32",
    )
    with pytest.raises(MalformedGenerateResponseError, match="compact logprob buffer"):
        await request(data, sampling_params={"logprobs": 2})


@pytest.mark.asyncio
async def test_replay_data_absence_is_not_treated_as_disabled_feature():
    with pytest.raises(
        MalformedGenerateResponseError, match="omitted requested routed_experts"
    ):
        await request(
            backend_options={
                "return_routed_experts": True,
                "routed_experts_shape": [1, 1],
            }
        )


def test_native_media_collapses_only_expanded_placeholder_spans():
    from renderers.backends.sglang import _unexpanded_prompt

    sidecar = MultiModalData(
        mm_placeholders={
            "image": [
                PlaceholderRange(offset=1, length=3),
                PlaceholderRange(offset=6, length=2),
            ]
        }
    )
    # A sampled assistant prefix and all template tokens outside media survive.
    assert _unexpanded_prompt([1, 2, 2, 2, 8, 9, 2, 2, 3], sidecar) == [
        1,
        2,
        8,
        9,
        2,
        3,
    ]
    with pytest.raises(ValueError, match="repeated placeholder"):
        _unexpanded_prompt([1, 2, 7, 2, 8, 9, 2, 2, 3], sidecar)


@pytest.mark.asyncio
async def test_prefill_versions_remain_prompt_relative():
    data = payload()
    spans = [
        {"version": "cached-v1", "start": 0, "end": 2},
        {"version": "v2", "start": 2, "end": 3},
    ]
    data["meta_info"]["prefill_weight_versions"] = spans
    result, _ = await request(data)
    assert result["prefill_weight_versions"] == spans
    data["meta_info"]["prefill_weight_versions"] = spans[:1]
    with pytest.raises(MalformedGenerateResponseError, match="cover prefill"):
        await request(data)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [None, "discovery", "worker", "timeout"])
async def test_router_cancel_discovers_and_aborts_each_worker(failure):
    from renderers.backends import EngineAbortError

    requests = []
    started = asyncio.Event()
    terminal = asyncio.Event()

    async def handler(req):
        requests.append(req)
        if req.url.path == "/generate":
            started.set()
            await terminal.wait()
        if req.url.path == "/workers":
            if failure == "discovery":
                return httpx.Response(503, json={"error": "unavailable"})
            return httpx.Response(
                200,
                json={
                    "workers": [{"url": "http://worker-1"}, {"url": "http://worker-2"}]
                },
            )
        if failure == "worker" and req.url.host == "worker-2":
            return httpx.Response(503, json={"error": "unavailable"})
        if failure == "timeout":
            await asyncio.Event().wait()
        terminal.set()
        return httpx.Response(200, json={})

    async with AsyncOpenAI(
        base_url="http://router/v1",
        api_key="test",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    ) as client:
        task = asyncio.create_task(
            generate(
                client=client,
                renderer=Renderer(),
                model="test",
                messages=[],
                backend="sglang",
                max_prompt_len=100,
                backend_options={"abort_strategy": "workers", "abort_timeout": 0.1},
            )
        )
        await started.wait()
        task.cancel()
        with pytest.raises(EngineAbortError if failure else asyncio.CancelledError):
            await task
    if failure != "discovery":
        aborts = [req for req in requests if req.url.path == "/abort_request"]
        assert {req.url.host for req in aborts} == {"worker-1", "worker-2"}
        rid = json.loads(requests[0].content)["rid"]
        assert all(json.loads(req.content) == {"rid": rid} for req in aborts)
        assert not any(req.url.host == "router" for req in aborts)


@pytest.mark.asyncio
async def test_training_sampling_constraints_pin_defaults_and_reject_overrides():
    options = {"sampling_constraints": {"temperature": 0.8, "top_k": -1, "min_p": 0.0}}
    _, requests = await request(backend_options=options)
    sampling = json.loads(requests[0].content)["sampling_params"]
    assert sampling["temperature"] == 0.8 and sampling["top_k"] == -1
    with pytest.raises(ValueError, match="violates training constraint"):
        await request(backend_options=options, sampling_params={"temperature": 0.4})
    with pytest.raises(ValueError, match="below the training requirement"):
        await request(
            backend_options={"min_top_logprobs": 8}, sampling_params={"logprobs": 4}
        )


@pytest.mark.asyncio
async def test_abort_acknowledgement_is_not_generation_termination():
    started = asyncio.Event()
    posts = []

    async def handler(req):
        posts.append(req.url.path)
        if req.url.path == "/generate":
            started.set()
            await asyncio.Event().wait()
        # The engine acknowledges enqueueing an abort but never returns the
        # original generation. This must fail drain, not claim cancellation.
        return httpx.Response(200, json={})

    async with AsyncOpenAI(
        base_url="http://engine/v1",
        api_key="test",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    ) as client:
        pending = asyncio.create_task(
            generate(
                client=client,
                renderer=Renderer(),
                model="test",
                messages=[],
                backend="sglang",
                max_prompt_len=100,
                backend_options={"abort_timeout": 0.05},
            )
        )
        await started.wait()
        pending.cancel()
        with pytest.raises(EngineAbortError, match="terminal engine reply"):
            await pending
    assert posts == ["/generate", "/abort_request"]
