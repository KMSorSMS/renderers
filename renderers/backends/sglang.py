"""Native SGLang token generation, without importing the SGLang runtime.

Protocol reference: GenerateReqInput and TokenizerManager in SGLang. The
normalized result is an internal token-evidence record, not an emulated HTTP
vLLM server. Rendering and response parsing remain in renderers.client.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import io
import logging
import math
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from openai import APIConnectionError, APIStatusError, APITimeoutError

from renderers.backends.base import EngineAbortedError, EngineAbortError, SGLangOptions
from renderers.client import (
    MalformedGenerateResponseError,
    OverlongPromptError,
    _parse_sampling_mask,
    _select_sampler_head,
)

logger = logging.getLogger(__name__)
_ALIASES = {
    "max_tokens": "max_new_tokens",
    "min_tokens": "min_new_tokens",
    "seed": "sampling_seed",
}
_SAMPLING_FIELDS = {
    "max_new_tokens",
    "min_new_tokens",
    "stop",
    "stop_token_ids",
    "stop_regex",
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "frequency_penalty",
    "presence_penalty",
    "repetition_penalty",
    "n",
    "json_schema",
    "regex",
    "ebnf",
    "structural_tag",
    "ignore_eos",
    "skip_special_tokens",
    "spaces_between_special_tokens",
    "no_stop_trim",
    "logit_bias",
    "sampling_seed",
    "custom_params",
    "watermark",
}


def _sampling(raw, stop_ids, prompt_len, options):
    values = {key: value for key, value in raw.items() if value is not None}
    values.pop("logprobs", None)
    replay_start = values.pop(
        "routed_experts_prompt_start", options.routed_experts_start_len
    )
    for old, new in _ALIASES.items():
        if old in values:
            if new in values and values[new] != values[old]:
                raise ValueError(f"Conflicting sampling parameters: {old} and {new}")
            values[new] = values.pop(old)
    unknown = set(values) - _SAMPLING_FIELDS
    if unknown:
        raise ValueError(f"Unsupported SGLang sampling parameters: {sorted(unknown)}")
    if values.get("n", 1) != 1:
        raise ValueError(
            "generate returns one completion; run independent requests for n > 1"
        )
    if values.get("no_stop_trim") is False:
        raise ValueError("Token-exact training requires no_stop_trim=True")
    values["stop_token_ids"] = list(
        dict.fromkeys(
            [
                *stop_ids,
                *(values.get("stop_token_ids") or []),
                *(options.stop_token_ids or []),
            ]
        )
    )
    if options.stop is not None:
        stops = [options.stop] if isinstance(options.stop, str) else options.stop
        requested = values.get("stop") or []
        requested = [requested] if isinstance(requested, str) else requested
        values["stop"] = list(dict.fromkeys([*requested, *stops]))
    if options.skip_special_tokens is not None:
        values["skip_special_tokens"] = options.skip_special_tokens
    values.setdefault("skip_special_tokens", False)
    values.setdefault("spaces_between_special_tokens", False)
    values["no_stop_trim"] = True
    values["n"] = 1
    limits = [
        value
        for value in (values.get("max_new_tokens"), options.max_response_tokens)
        if value is not None
    ]
    if options.max_context_len is not None:
        budget = options.max_context_len - prompt_len
        if budget <= 0:
            raise OverlongPromptError(
                prompt_len=prompt_len, max_prompt_len=options.max_context_len
            )
        limits.append(budget)
    if limits:
        values["max_new_tokens"] = min(limits)
    if type(replay_start) is not int or not 0 <= replay_start <= prompt_len:
        raise ValueError("routed_experts_prompt_start must lie inside the prompt")
    for key, expected in options.sampling_constraints.items():
        key = _ALIASES.get(key, key)
        if key not in _SAMPLING_FIELDS:
            raise ValueError(f"Unknown SGLang sampling constraint: {key}")
        if key in values and values[key] != expected:
            raise ValueError(
                f"SGLang sampling {key}={values[key]!r} violates training constraint {expected!r}"
            )
        values[key] = expected
    return values, replay_start


def _media_source(source, modality):
    if isinstance(source, Mapping):
        source = source.get("url")
    if isinstance(source, Path) or (
        isinstance(source, str)
        and len(source) < 1024
        and not source.startswith(("data:", "http://", "https://"))
        and Path(source).is_file()
    ):
        source = Path(source).read_bytes()
    if isinstance(source, bytes):
        return base64.b64encode(source).decode("ascii")
    if isinstance(source, str) and source:
        return source
    # Pillow is optional; avoid importing it for text-only workers.
    if modality == "image" and hasattr(source, "save"):
        buffer = io.BytesIO()
        source.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")
    raise ValueError(
        f"SGLang {modality} input requires a URL, file, bytes, or supported image"
    )


def _media_inputs(messages, renderer):
    order = getattr(renderer, "messages_in_prompt_order", None)
    if order is not None:
        messages = order(messages)
    result: dict[str, list] = {}
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, Mapping):
                continue
            kind = part.get("type", "")
            modality = kind.removesuffix("_url")
            if modality not in {"image", "video", "audio"}:
                continue
            source = part.get(kind)
            if source is None:
                source = part.get("url", part.get("path", part.get(modality)))
            result.setdefault(f"{modality}_data", []).append(
                _media_source(source, modality)
            )
    return result


def _unexpanded_prompt(prompt_ids, mm_data):
    """Raw media processors expand one placeholder per item themselves.

    Collapse only renderer-declared repeated placeholder ids. Message/template
    tokens outside those ranges are copied byte-for-byte, including sampled
    history. The response is checked against the expanded renderer prompt.
    """
    spans = sorted(
        (span.offset, span.length)
        for group in mm_data.mm_placeholders.values()
        for span in group
    )
    if not spans:
        raise ValueError("SGLang raw media requires renderer placeholder ranges")
    result, cursor = [], 0
    for offset, length in spans:
        if length < 1 or offset < cursor or offset + length > len(prompt_ids):
            raise ValueError("Invalid multimodal placeholder range")
        tokens = prompt_ids[offset : offset + length]
        if any(token != tokens[0] for token in tokens):
            raise ValueError(
                "SGLang raw media requires repeated placeholder ids; this renderer needs a model-specific encoder"
            )
        result.extend(prompt_ids[cursor:offset])
        result.append(tokens[0])
        cursor = offset + length
    result.extend(prompt_ids[cursor:])
    return result


def _check_prompt(data, prompt_ids, mm_data):
    actual = data.get("prompt_token_ids")
    meta = data["meta_info"]
    count = meta.get("prompt_tokens", len(prompt_ids))
    if type(count) is not int or count != len(prompt_ids):
        raise MalformedGenerateResponseError(
            "SGLang prompt token count differs from rendered input"
        )
    if actual is None:
        if mm_data and not mm_data.is_empty():
            raise MalformedGenerateResponseError(
                "Multimodal SGLang responses must include prompt_token_ids"
            )
        return
    if (
        not isinstance(actual, list)
        or len(actual) != len(prompt_ids)
        or any(type(token) is not int or token < 0 for token in actual)
    ):
        raise MalformedGenerateResponseError(
            "SGLang changed the rendered prompt length"
        )
    # SGLang may substitute cache-identity pad values inside multimodal spans.
    # These do not represent vocabulary tokens; retain the renderer's model ids.
    placeholders = set()
    if mm_data:
        for spans in mm_data.mm_placeholders.values():
            for span in spans:
                placeholders.update(range(span.offset, span.offset + span.length))
    if any(
        a != b and i not in placeholders
        for i, (a, b) in enumerate(zip(actual, prompt_ids, strict=True))
    ):
        raise MalformedGenerateResponseError("SGLang changed rendered text tokens")


async def _abort_targets(client, endpoint, headers, options):
    if options.abort_urls:
        return list(dict.fromkeys(url.rstrip("/") for url in options.abort_urls))
    if options.abort_strategy == "endpoint":
        return [endpoint]
    response = await client.get(
        f"{endpoint}/workers",
        cast_to=httpx.Response,
        options={"headers": headers or {}},
    )
    data = response.json()
    workers = data.get("workers") if isinstance(data, Mapping) else None
    if not isinstance(workers, list) or not workers:
        raise ValueError("Router worker discovery returned no engine workers")
    urls = [
        worker.get("url") if isinstance(worker, Mapping) else worker
        for worker in workers
    ]
    if any(
        not isinstance(url, str) or not url.startswith(("http://", "https://"))
        for url in urls
    ):
        raise ValueError("Router worker discovery returned invalid engine URLs")
    return list(dict.fromkeys(url.rstrip("/") for url in urls))


async def _abort(client, endpoint, rid, headers, options):
    async def perform():
        targets = await _abort_targets(client, endpoint, headers, options)
        results = await asyncio.gather(
            *(
                client.post(
                    f"{url}/abort_request",
                    cast_to=httpx.Response,
                    body={"rid": rid},
                    options={"headers": headers or {}},
                )
                for url in targets
            ),
            return_exceptions=True,
        )
        failed = [
            url
            for url, result in zip(targets, results, strict=True)
            if isinstance(result, BaseException)
        ]
        if failed:
            raise RuntimeError(f"Abort failed on engine workers: {failed}")

    try:
        await asyncio.wait_for(perform(), timeout=options.abort_timeout)
    except Exception as exc:
        raise EngineAbortError(
            f"Could not confirm abort of SGLang request {rid}"
        ) from exc


async def _abort_and_drain(
    client, endpoint, rid, headers, options, request_task, *, transport_lost
):
    """An abort acknowledgement is not completion; retain the original reply."""
    try:
        if not request_task.done() or transport_lost:
            await _abort(client, endpoint, rid, headers, options)
        if transport_lost:
            raise EngineAbortError(
                f"Lost the terminal response for SGLang request {rid}"
            )
        try:
            await asyncio.wait_for(
                asyncio.shield(request_task), timeout=options.abort_timeout
            )
        except APIStatusError as exc:
            # SGLang's native /generate returns 400 when an abort completes.
            # A gateway timeout/5xx is not evidence of engine termination.
            if exc.status_code != 400:
                raise EngineAbortError(
                    f"No confirmed terminal engine reply for {rid}"
                ) from exc
        except (TimeoutError, APITimeoutError, APIConnectionError) as exc:
            raise EngineAbortError(
                f"No confirmed terminal engine reply for {rid}"
            ) from exc
    finally:
        if not request_task.done():
            request_task.cancel()
            await asyncio.gather(request_task, return_exceptions=True)


def _request_features(body, *, options, head_size, replay_start):
    if options.lora_path is not None:
        body["lora_path"] = options.lora_path
    if head_size is not None:
        body["top_logprobs_num"] = head_size + 2
        if options.compact_logprobs:
            body.update(
                # Older SGLang releases require this flag for base64 even when
                # they do not yet support compact output top logprobs. Those
                # releases return the ordinary output arrays, also supported.
                return_flat_raw_top_logprobs=True,
                return_flat_raw_output_top_logprobs=True,
                return_flat_raw_top_logprobs_b64=True,
            )
    if options.return_sampling_mask:
        body.update(
            return_sampling_mask=True,
            sampling_logprobs_mode=options.sampling_logprobs_mode,
        )
    if options.return_routed_experts:
        body.update(return_routed_experts=True, routed_experts_start_len=replay_start)
    if options.return_indexer_topk:
        body["return_indexer_topk"] = True


def _request_routing(body, *, extra_headers, options, cache_salt, priority):
    if cache_salt is not None:
        body["extra_key"] = cache_salt
    if priority is not None:
        body["priority"] = priority
    headers = dict(extra_headers or {})
    session_id = next(
        (value for key, value in headers.items() if key.lower() == "x-session-id"), None
    )
    if session_id:
        if options.routing_header:
            headers[options.routing_header] = session_id
        body["session_id"] = session_id
        if options.routing_key_from_session:
            body["routing_key"] = session_id
    return headers


def _request_media(
    body, *, messages, renderer, mm_data, prompt_ids, process_multimodal
):
    media = _media_inputs(messages, renderer)
    if media:
        if not process_multimodal or not mm_data or mm_data.is_empty():
            raise ValueError(
                "SGLang multimodal generation requires processed renderer tokens and sidecar"
            )
        body.update(media)
        body["input_ids"] = _unexpanded_prompt(prompt_ids, mm_data)
        body["return_prompt_token_ids"] = True
    elif mm_data and not mm_data.is_empty():
        raise ValueError(
            "SGLang multimodal sidecar requires the original media in messages"
        )


def _validate_request_options(options, head_size, is_initial_turn, prompt_len):
    if options.min_top_logprobs is not None and (
        head_size is None or head_size < options.min_top_logprobs
    ):
        raise ValueError("Requested top logprobs are below the training requirement")
    if (
        head_size is not None
        and options.return_sampling_mask
        and options.sampling_logprobs_mode == "selected"
    ):
        raise ValueError(
            "Top-logprob heads with behavior sampling require sampling_logprobs_mode='support'"
        )
    if (
        is_initial_turn
        and options.max_prompt_len is not None
        and prompt_len > options.max_prompt_len
    ):
        raise OverlongPromptError(
            prompt_len=prompt_len, max_prompt_len=options.max_prompt_len
        )


async def generate_raw(
    *,
    client,
    renderer,
    messages,
    model,
    prompt_ids,
    stop_token_ids,
    sampling_params,
    head_size,
    mm_data,
    process_multimodal,
    cache_salt,
    priority,
    extra_headers,
    backend_options,
    is_initial_turn=True,
):
    options = SGLangOptions.model_validate(backend_options or {})
    _validate_request_options(options, head_size, is_initial_turn, len(prompt_ids))
    sp, replay_start = _sampling(
        sampling_params, stop_token_ids, len(prompt_ids), options
    )
    rid = uuid.uuid4().hex
    body: dict[str, Any] = {
        "input_ids": prompt_ids,
        "sampling_params": sp,
        "rid": rid,
        "return_logprob": True,
        "return_text_in_logprobs": False,
        "stream": False,
    }
    _request_features(
        body, options=options, head_size=head_size, replay_start=replay_start
    )
    headers = _request_routing(
        body,
        extra_headers=extra_headers,
        options=options,
        cache_salt=cache_salt,
        priority=priority,
    )
    _request_media(
        body,
        messages=messages,
        renderer=renderer,
        mm_data=mm_data,
        prompt_ids=prompt_ids,
        process_multimodal=process_multimodal,
    )
    base = str(client.base_url).rstrip("/").removesuffix("/v1")
    # A retry can duplicate a stochastic generation. The caller owns retry policy.
    transport = (
        client.with_options(max_retries=0)
        if hasattr(client, "with_options")
        else client
    )
    request_task = asyncio.create_task(
        transport.post(
            f"{base}/generate",
            cast_to=httpx.Response,
            body=body,
            options={"headers": headers},
        )
    )
    try:
        response = await asyncio.shield(request_task)
    except (
        asyncio.CancelledError,
        TimeoutError,
        APITimeoutError,
        APIConnectionError,
    ) as exc:
        cleanup = asyncio.create_task(
            _abort_and_drain(
                transport,
                base,
                rid,
                headers,
                options,
                request_task,
                transport_lost=not isinstance(exc, asyncio.CancelledError),
            )
        )
        try:
            # Repeated cancellation must not interrupt the bounded cleanup.
            while not cleanup.done():
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    continue
            cleanup.result()
        except EngineAbortError as abort_error:
            raise abort_error from exc
        raise
    try:
        data = response.json()
    except ValueError as exc:
        raise MalformedGenerateResponseError("SGLang returned invalid JSON") from exc
    if not isinstance(data, Mapping) or not isinstance(data.get("meta_info"), Mapping):
        raise MalformedGenerateResponseError(
            "SGLang must return one object with meta_info"
        )
    normalized = _normalize(
        data, rid, len(prompt_ids), head_size, options, replay_start
    )
    _check_prompt(data, prompt_ids, mm_data)
    if mm_data and not mm_data.is_empty():
        normalized["mm_placeholders"] = {
            kind: [{"offset": span.offset, "length": span.length} for span in spans]
            for kind, spans in mm_data.mm_placeholders.items()
        }
    return normalized, False


def _entry(value):
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        raise MalformedGenerateResponseError(
            "SGLang logprob entry must be [logprob, token_id, text]"
        )
    probability, token = value[:2]
    if (
        type(token) is not int
        or token < 0
        or isinstance(probability, bool)
        or not isinstance(probability, (float, int))
    ):
        raise MalformedGenerateResponseError(
            "SGLang logprob entry has invalid token/probability types"
        )
    try:
        probability = float(probability)
    except OverflowError as exc:
        raise MalformedGenerateResponseError(
            "SGLang logprob is outside float32 range"
        ) from exc
    if (
        not math.isfinite(probability)
        or probability < -float(np.finfo(np.float32).max)
        or probability > 1e-6
    ):
        raise MalformedGenerateResponseError(
            "SGLang logprob must be finite and non-positive"
        )
    return {"token_id": token, "logprob": float(probability)}


def _flat_top(meta, count):
    shape = meta.get("output_top_logprobs_shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or any(type(x) is not int or x < 0 for x in shape)
    ):
        raise MalformedGenerateResponseError("Invalid SGLang compact logprob shape")
    if shape[0] != count or meta.get("output_top_logprobs_null_prefix", 0) != 0:
        raise MalformedGenerateResponseError(
            "SGLang compact logprobs do not cover every completion token"
        )
    arrays = []
    for name, dtype in (("idx", "int32"), ("val", "float32")):
        prefix = f"output_top_logprobs_{name}_flat"
        try:
            if prefix + "_b64" in meta:
                if meta.get(prefix + "_b64_dtype") != dtype:
                    raise ValueError("unexpected dtype")
                array = np.frombuffer(
                    base64.b64decode(meta[prefix + "_b64"], validate=True),
                    dtype=np.dtype(dtype).newbyteorder("<"),
                )
            else:
                array = np.asarray(meta[prefix])
                if name == "idx" and array.dtype.kind not in "iu":
                    raise ValueError("token ids must be integers")
            if array.size != math.prod(shape):
                raise ValueError("buffer length mismatch")
            arrays.append(array.reshape(shape))
        except (KeyError, ValueError, TypeError, binascii.Error) as exc:
            raise MalformedGenerateResponseError(
                "Invalid SGLang compact logprob buffer"
            ) from exc
    ids, probabilities = arrays
    return [
        [_entry((float(p), int(i))) for i, p in zip(id_row, p_row, strict=True)]
        for id_row, p_row in zip(ids, probabilities, strict=True)
    ]


def _replay_array(meta, name, shape, start, end):
    value = meta.get(name)
    if value is None:
        raise MalformedGenerateResponseError(f"SGLang omitted requested {name}")
    if shape is None:
        raise ValueError(f"{name} requires its layer/top-k shape")
    full_shape = [max(0, end - start), *shape]
    try:
        raw = base64.b64decode(value, validate=True)
    except (ValueError, TypeError, binascii.Error) as exc:
        raise MalformedGenerateResponseError(f"Invalid {name} base64") from exc
    if len(raw) != math.prod(full_shape) * 4:
        raise MalformedGenerateResponseError(
            f"{name} bytes do not match configured shape {full_shape}"
        )
    return {"data": value, "shape": full_shape, "dtype": "int32", "start": start}


def _select_heads(choice, entries, token_ids, head_size):
    supports = _parse_sampling_mask(choice, token_ids)
    top_ids, top_values = [], []
    for index, entry in enumerate(entries):
        candidates = entry["top_logprobs"]
        head = {candidate["token_id"]: candidate["logprob"] for candidate in candidates}
        if len(head) != len(candidates):
            raise MalformedGenerateResponseError(
                "Duplicate SGLang top logprob token ids"
            )
        ids, values = _select_sampler_head(
            head,
            entry["token_id"],
            entry["logprob"],
            head_size,
            supports[index] if supports is not None else None,
            missing_logprob=None,
        )
        top_ids.append(ids)
        top_values.append(values)
    choice["completion_top_ids"] = top_ids
    choice["completion_top_logprobs"] = top_values


def _normalize(data, rid, prompt_len, head_size, options, replay_start):
    meta = data["meta_info"]
    reason = meta.get("finish_reason")
    reason = reason.get("type") if isinstance(reason, Mapping) else reason
    if reason == "abort":
        raise EngineAbortedError(
            f"SGLang aborted request {meta.get('id', rid)}: {meta.get('finish_reason')}"
        )
    if reason not in {"stop", "length", "content_filter"}:
        raise MalformedGenerateResponseError(
            f"Unknown SGLang finish reason: {reason!r}"
        )
    rows = meta.get("output_token_logprobs")
    if not isinstance(rows, list):
        raise MalformedGenerateResponseError("SGLang omitted sampled-token logprobs")
    entries = [_entry(row) for row in rows]
    token_ids = [entry["token_id"] for entry in entries]
    if "output_ids" in data and (
        not isinstance(data["output_ids"], list)
        or any(type(token) is not int or token < 0 for token in data["output_ids"])
        or data["output_ids"] != token_ids
    ):
        raise MalformedGenerateResponseError(
            "SGLang output_ids and sampled-token evidence disagree"
        )
    count = meta.get("completion_tokens", len(token_ids))
    if type(count) is not int or count != len(token_ids):
        raise MalformedGenerateResponseError(
            "SGLang completion token count does not match logprobs"
        )
    choice = {
        "token_ids": token_ids,
        "logprobs": {"content": entries},
        "finish_reason": reason,
    }
    if head_size is not None:
        if "output_top_logprobs_shape" in meta:
            heads = _flat_top(meta, len(token_ids))
        else:
            rows = meta.get("output_top_logprobs")
            if not isinstance(rows, list) or len(rows) != len(token_ids):
                raise MalformedGenerateResponseError(
                    "SGLang omitted requested top logprobs"
                )
            heads = [[_entry(value) for value in row] for row in rows]
        for entry, head in zip(entries, heads, strict=True):
            entry["top_logprobs"] = head
    if options.return_sampling_mask:
        _sampling_evidence(meta, choice, entries, options)
    choice["completion_logprobs_array"] = np.asarray(
        [entry["logprob"] for entry in entries], dtype=np.float32
    )
    if head_size is not None:
        _select_heads(choice, entries, token_ids, head_size)
    if options.return_routed_experts:
        choice["routed_experts"] = _replay_array(
            meta,
            "routed_experts",
            options.routed_experts_shape,
            replay_start,
            prompt_len + len(token_ids) - 1,
        )
    if options.return_indexer_topk:
        choice["indexer_topk"] = _replay_array(
            meta,
            "indexer_topk",
            options.indexer_topk_shape,
            0,
            prompt_len + len(token_ids) - 1,
        )
    spans = meta.get("weight_versions")
    if spans is None and meta.get("weight_version") is not None and token_ids:
        spans = [
            {"version": str(meta["weight_version"]), "start": 0, "end": len(token_ids)}
        ]
    choice["weight_versions"] = _weight_versions(spans, len(token_ids))
    choice["prefill_weight_versions"] = _weight_versions(
        meta.get("prefill_weight_versions"), prompt_len, kind="prefill"
    )
    return {
        "request_id": meta.get("id") or data.get("request_id") or rid,
        "choices": [choice],
        "usage": {
            "prompt_tokens": prompt_len,
            "completion_tokens": len(token_ids),
            "total_tokens": prompt_len + len(token_ids),
        },
    }


def _weight_versions(spans, count, *, kind="completion"):
    if spans is None:
        return None
    if not isinstance(spans, list):
        raise MalformedGenerateResponseError("Invalid SGLang weight version spans")
    cursor = 0
    for span in spans:
        if (
            not isinstance(span, Mapping)
            or not isinstance(span.get("version"), str)
            or type(span.get("start")) is not int
            or type(span.get("end")) is not int
            or span["start"] != cursor
            or not cursor <= span["end"] <= count
        ):
            raise MalformedGenerateResponseError(
                "Invalid SGLang weight version span coverage"
            )
        cursor = span["end"]
    if cursor != count:
        raise MalformedGenerateResponseError(
            f"SGLang weight version spans do not cover {kind}"
        )
    return spans


def _sampling_evidence(meta, choice, entries, options):
    masks = meta.get("output_token_sampling_mask")
    values = meta.get("output_token_sampling_logprobs")
    if (
        not isinstance(masks, list)
        or not isinstance(values, list)
        or len(masks) != len(entries)
        or len(values) != len(entries)
    ):
        raise MalformedGenerateResponseError(
            "SGLang omitted requested behavior sampling evidence"
        )
    for entry, support, probabilities in zip(entries, masks, values, strict=True):
        if not isinstance(support, list) or entry["token_id"] not in support:
            raise MalformedGenerateResponseError(
                "Sampled token is missing from sampling support"
            )
        if options.sampling_logprobs_mode == "support":
            if not isinstance(probabilities, list) or len(probabilities) != len(
                support
            ):
                raise MalformedGenerateResponseError(
                    "Behavior probabilities do not cover sampling support"
                )
            head = [_entry((p, i)) for i, p in zip(support, probabilities, strict=True)]
            if not math.isclose(
                math.fsum(math.exp(item["logprob"]) for item in head), 1.0, abs_tol=1e-4
            ):
                raise MalformedGenerateResponseError(
                    "Behavior probabilities must normalize over sampling support"
                )
            entry["top_logprobs"] = head
            entry["logprob"] = next(
                item["logprob"]
                for item in head
                if item["token_id"] == entry["token_id"]
            )
        else:
            entry["logprob"] = _entry((probabilities, entry["token_id"]))["logprob"]
            if "top_logprobs" in entry:
                raise ValueError(
                    "Top-logprob heads with behavior sampling require sampling_logprobs_mode='support'"
                )
    choice["sampling_mask"] = masks
    _parse_sampling_mask(choice, [entry["token_id"] for entry in entries])
