"""Configuration shared by native inference backends.

The backend boundary is below rendering: input ids are already rendered and
responses carry token ids and sampling evidence, never re-tokenized text.
"""

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    PositiveInt,
    model_validator,
)


class SGLangOptions(BaseModel):
    """Serializable options for the native ``/generate`` endpoint.

    Shapes describe the two trailing axes of SGLang's headerless int32 replay
    buffers. They must match the running engine, not the training partition.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    lora_path: str | None = None
    max_prompt_len: PositiveInt | None = None
    stop: str | list[str] | None = None
    stop_token_ids: list[int] | None = None
    skip_special_tokens: bool | None = None
    routing_header: str | None = None
    sampling_constraints: dict[str, JsonValue] = Field(default_factory=dict)
    min_top_logprobs: PositiveInt | None = None
    compact_logprobs: bool = True
    return_routed_experts: bool = False
    routed_experts_shape: tuple[PositiveInt, PositiveInt] | None = None
    routed_experts_start_len: int = Field(default=0, ge=0)
    return_indexer_topk: bool = False
    indexer_topk_shape: tuple[PositiveInt, PositiveInt] | None = None
    return_sampling_mask: bool = False
    sampling_logprobs_mode: Literal["selected", "support"] = "support"
    routing_key_from_session: bool = True
    abort_strategy: Literal["endpoint", "workers"] = "endpoint"
    abort_urls: list[str] | None = Field(default=None, min_length=1)
    abort_timeout: float = Field(default=5.0, gt=0)
    max_response_tokens: PositiveInt | None = None
    max_context_len: PositiveInt | None = None

    @model_validator(mode="after")
    def _require_replay_shapes(self):
        if self.abort_urls and any(
            not url.startswith(("http://", "https://")) for url in self.abort_urls
        ):
            raise ValueError("abort_urls must contain absolute HTTP(S) engine URLs")
        for enabled, shape in (
            (self.return_routed_experts, self.routed_experts_shape),
            (self.return_indexer_topk, self.indexer_topk_shape),
        ):
            if enabled and shape is None:
                raise ValueError(
                    "Requested replay buffers require their layer/top-k shape"
                )
        return self


class EngineAbortedError(RuntimeError):
    """The engine explicitly aborted a generation; it is not a successful stop."""


class EngineAbortError(RuntimeError):
    """Abort delivery or the abandoned generation's terminal reply was unconfirmed."""
