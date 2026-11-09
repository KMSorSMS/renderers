"""vLLM native generation protocol and multimodal serialization."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import httpx

from renderers.base import Message, MultiModalData, Renderer, _require_transformers


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
    """Return the native payload; common token validation lives in client.py."""
    if backend_options:
        raise ValueError("vLLM does not accept SGLang backend_options")
    sp = dict(sampling_params)
    sp["stop_token_ids"] = stop_token_ids
    sp["logprobs"] = head_size + 2 if head_size is not None else 1
    if head_size is not None:
        sp["extra_args"] = {
            "prl_compact_logprobs": True,
            **(sp.get("extra_args") or {}),
        }
    sp.setdefault("skip_special_tokens", False)
    body = {"model": model, "token_ids": prompt_ids, "sampling_params": sp}
    content_parts = (
        _content_parts(messages, renderer=renderer) if not process_multimodal else None
    )
    if content_parts:
        body.update(content_parts=content_parts, return_token_ids=True)
    if process_multimodal and mm_data and not mm_data.is_empty():
        features = _build_mm_features(renderer, mm_data)
        if features is not None:
            body["features"] = features
    if cache_salt is not None:
        body["cache_salt"] = cache_salt
    if priority is not None:
        body["priority"] = priority
    base = str(client.base_url).rstrip("/").removesuffix("/v1")
    kwargs = {"cast_to": httpx.Response, "body": body}
    if extra_headers:
        kwargs["options"] = cast(Any, {"headers": extra_headers})
    response = await client.post(f"{base}/inference/v1/generate", **kwargs)
    # Imported here to preserve the public client.py parser API without a cycle.
    from renderers.client import parse_generate_response

    return parse_generate_response(response.content), bool(content_parts)


_MEDIA_URL_TYPES = {
    "image": "image_url",
    "image_url": "image_url",
    "audio": "audio_url",
    "audio_url": "audio_url",
    "video": "video_url",
    "video_url": "video_url",
}


def _content_parts(
    messages: list[Message], *, renderer: Renderer
) -> list[dict[str, Any]]:
    """Flatten raw media in prompt order for vLLM's token generate endpoint."""
    # Some templates reorder tool results. Use the same order for raw media,
    # including when the caller supplies a prebuilt prompt and skips rendering.
    order_messages = getattr(renderer, "messages_in_prompt_order", None)
    if order_messages is not None:
        messages = order_messages(messages)
    parts: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, Mapping):
                continue
            part_type = part.get("type")
            if part_type is None:
                part_type = next(
                    (key for key in _MEDIA_URL_TYPES if part.get(key)), None
                )
            if not isinstance(part_type, str) or part_type not in _MEDIA_URL_TYPES:
                continue
            source = part.get(part_type)
            if source is None:
                source = part.get(_MEDIA_URL_TYPES[part_type]) or part.get("url")
            url = source.get("url") if isinstance(source, Mapping) else source
            if not isinstance(url, str) or not url:
                raise ValueError(f"{part_type} content part is missing a URL")
            parts.append({"type": _MEDIA_URL_TYPES[part_type], "url": url})
    return parts


def _build_mm_features(
    renderer: Renderer,
    mm_data: MultiModalData,
) -> dict[str, Any] | None:
    """Serialize ``MultiModalData`` to vLLM's ``/inference/v1/generate`` features payload.

    vLLM's ``MultiModalFeatures`` carries three things: hashes (for cache
    lookup), placeholder positions (so the engine knows where in the
    token stream each item lives), and per-item ``MultiModalKwargsItem``
    base64-encoded. The encoding requires vLLM-side type info — what
    fields belong to each modality, how they batch — and is currently
    model-family specific. For now we dispatch on the renderer class;
    extend the dispatch table as more multimodal renderers land.

    This serializer is intentionally isolated in the vLLM backend. SGLang uses
    native media fields and does not import vLLM or encode its feature schema.
    """
    from renderers.gemma4 import Gemma4Renderer
    from renderers.nemotron3 import Nemotron35Renderer
    from renderers.qwen3_vl import Qwen3VLRenderer
    from renderers.qwen35 import Qwen35Renderer

    renderer_cls = type(renderer)

    # Qwen3-VL and Qwen3.5 both ship ``pixel_values`` + ``image_grid_thw``
    # via the shared Qwen2-VL field factory. ``spatial_merge_size=2`` is
    # the family default and matches every Qwen-VL processor in tree.
    if issubclass(renderer_cls, (Qwen3VLRenderer, Qwen35Renderer)):
        return _build_qwen_vl_features(mm_data, spatial_merge_size=2)
    if issubclass(renderer_cls, Gemma4Renderer):
        return _build_gemma4_features(mm_data)
    if issubclass(renderer_cls, Nemotron35Renderer):
        return _build_nemotron35_features(mm_data)

    raise NotImplementedError(
        f"Multimodal serialization not implemented for {renderer_cls.__name__}. "
        "Add a dispatch branch in renderers.client._build_mm_features."
    )


def _build_nemotron35_features(mm_data: MultiModalData) -> dict[str, Any]:
    """Encode dynamic-resolution Nemotron image inputs for vLLM."""
    _require_transformers("Encoding Nemotron 3.5 multimodal features for vLLM")
    try:
        import torch
        from transformers.feature_extraction_utils import BatchFeature
        from vllm.entrypoints.scale_out.token_in_token_out.mm_serde import (
            encode_mm_kwargs_item,
        )
        from vllm.multimodal.inputs import (
            MultiModalFieldConfig,
            MultiModalKwargsItems,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Nemotron 3.5 multimodal generate requires vLLM and torch."
        ) from exc

    out: dict[str, Any] = {
        "mm_hashes": {},
        "mm_placeholders": {},
        "kwargs_data": {},
    }
    image_items = mm_data.mm_items.get("image") or []
    if image_items:
        pixel_values_flat = []
        imgs_sizes = []
        num_tokens_per_image = []
        for item in image_items:
            pixel_values = torch.as_tensor(item["pixel_values"])
            if pixel_values.ndim != 4 or pixel_values.shape[0] != 1:
                raise ValueError(
                    "Each Nemotron pixel_values item must have shape (1, C, H, W)."
                )
            sizes = item["imgs_sizes"]
            token_counts = item["num_tokens"]
            if not (len(sizes) == len(token_counts) == 1):
                raise ValueError(
                    "Each Nemotron image item must contain one size and token count."
                )
            pixel_values_flat.append(pixel_values[0])
            imgs_sizes.append(tuple(int(value) for value in sizes[0]))
            num_tokens_per_image.append(int(token_counts[0]))

        hf_inputs = BatchFeature(
            data={
                "pixel_values_flat": pixel_values_flat,
                "imgs_sizes": imgs_sizes,
                "num_tokens_per_image": num_tokens_per_image,
            }
        )
        field_config = {
            "pixel_values_flat": MultiModalFieldConfig.batched("image"),
            "num_tokens_per_image": MultiModalFieldConfig.batched(
                "image", keep_on_cpu=True
            ),
            "imgs_sizes": MultiModalFieldConfig.batched("image", keep_on_cpu=True),
        }
        kwargs_items = MultiModalKwargsItems.from_hf_inputs(hf_inputs, field_config)
        out["kwargs_data"]["image"] = [
            encode_mm_kwargs_item(item) for item in kwargs_items["image"]
        ]
        out["mm_hashes"]["image"] = list(mm_data.mm_hashes.get("image") or [])
        out["mm_placeholders"]["image"] = [
            {"offset": placeholder.offset, "length": placeholder.length}
            for placeholder in mm_data.mm_placeholders.get("image") or []
        ]

    if not any(out["kwargs_data"].values()):
        out["kwargs_data"] = None
    return out


def _build_gemma4_features(mm_data: MultiModalData) -> dict[str, Any]:
    """vLLM features payload for Gemma 4 image inputs.

    Hugging Face names the position field ``image_position_ids`` while
    vLLM's Gemma 4 processor schema calls it ``pixel_position_ids``. Keep
    renderer output faithful to the HF processor and translate at this
    engine-specific boundary.
    """
    _require_transformers("Encoding Gemma 4 multimodal features for vLLM")
    try:
        import torch
        from transformers.feature_extraction_utils import BatchFeature
        from vllm.entrypoints.scale_out.token_in_token_out.mm_serde import (
            encode_mm_kwargs_item,
        )
        from vllm.multimodal.inputs import (
            MultiModalFieldConfig,
            MultiModalKwargsItems,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Gemma 4 multimodal generate via /inference/v1/generate requires "
            "a vLLM release with Gemma 4 support and `torch`."
        ) from exc

    out: dict[str, Any] = {
        "mm_hashes": {},
        "mm_placeholders": {},
        "kwargs_data": {},
    }
    image_items = mm_data.mm_items.get("image") or []
    if image_items:
        pixel_values = torch.cat(
            [torch.as_tensor(item["pixel_values"]) for item in image_items], dim=0
        )
        pixel_position_ids = torch.cat(
            [torch.as_tensor(item["image_position_ids"]) for item in image_items],
            dim=0,
        )
        hf_inputs = BatchFeature(
            data={
                "pixel_values": pixel_values,
                "pixel_position_ids": pixel_position_ids,
            }
        )
        field_config = {
            "pixel_values": MultiModalFieldConfig.batched("image"),
            "pixel_position_ids": MultiModalFieldConfig.batched("image"),
        }
        kwargs_items = MultiModalKwargsItems.from_hf_inputs(hf_inputs, field_config)
        out["kwargs_data"]["image"] = [
            encode_mm_kwargs_item(item) for item in kwargs_items["image"]
        ]
        out["mm_hashes"]["image"] = list(mm_data.mm_hashes.get("image") or [])
        out["mm_placeholders"]["image"] = [
            {"offset": placeholder.offset, "length": placeholder.length}
            for placeholder in mm_data.mm_placeholders.get("image") or []
        ]

    if not any(out["kwargs_data"].values()):
        out["kwargs_data"] = None
    return out


def _build_qwen_vl_features(
    mm_data: MultiModalData, *, spatial_merge_size: int
) -> dict[str, Any]:
    """vLLM features payload for the Qwen-VL family (Qwen2-VL / Qwen3-VL).

    Stacks per-image processor outputs back into a batched ``BatchFeature``,
    runs the Qwen2-VL field factory (shared across the family), wraps as
    ``MultiModalKwargsItems``, base64-encodes each item, and assembles a
    JSON-serializable dict matching vLLM's ``MultiModalFeatures`` schema.

    Returns ``None`` semantics live one level up — this helper assumes
    the caller already verified ``mm_data`` is non-empty.
    """
    _require_transformers("Encoding Qwen-VL multimodal features for vLLM")
    try:
        import torch
        from transformers.feature_extraction_utils import BatchFeature
        from vllm.entrypoints.scale_out.token_in_token_out.mm_serde import (
            encode_mm_kwargs_item,
        )
        from vllm.model_executor.models.qwen2_vl import _create_qwen2vl_field_factory
        from vllm.multimodal.inputs import MultiModalKwargsItems
    except ImportError as exc:
        raise RuntimeError(
            "Multimodal generate via /inference/v1/generate requires `vllm` "
            "and `torch` to encode the features payload. Install vLLM in this "
            "environment, or pre-build features upstream."
        ) from exc

    out: dict[str, Any] = {
        "mm_hashes": {},
        "mm_placeholders": {},
        "kwargs_data": {},
    }

    image_items = mm_data.mm_items.get("image") or []
    if image_items:
        # mm_items now ship numpy arrays (the renderer is torch-free);
        # convert at this vLLM-glue boundary where torch is already a
        # hard dependency.
        pixel_values = torch.cat(
            [torch.as_tensor(it["pixel_values"]) for it in image_items], dim=0
        )
        image_grid_thw = torch.cat(
            [torch.as_tensor(it["image_grid_thw"]) for it in image_items], dim=0
        )
        hf_inputs = BatchFeature(
            data={"pixel_values": pixel_values, "image_grid_thw": image_grid_thw}
        )
        config = _create_qwen2vl_field_factory(spatial_merge_size)(hf_inputs)
        kwargs_items = MultiModalKwargsItems.from_hf_inputs(hf_inputs, config)
        encoded = [encode_mm_kwargs_item(it) for it in kwargs_items["image"]]
        out["kwargs_data"]["image"] = encoded
        out["mm_hashes"]["image"] = list(mm_data.mm_hashes.get("image") or [])
        out["mm_placeholders"]["image"] = [
            {"offset": p.offset, "length": p.length}
            for p in mm_data.mm_placeholders.get("image") or []
        ]

    # If kwargs_data is empty across all modalities, drop the key so vLLM
    # falls back to the hash-only (cache-hit) path. Otherwise hand it the
    # full payload.
    if not any(out["kwargs_data"].values()):
        out["kwargs_data"] = None

    return out
