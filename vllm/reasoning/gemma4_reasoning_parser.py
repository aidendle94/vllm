# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from typing import TYPE_CHECKING

from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.reasoning.basic_parsers import BaseThinkingReasoningParser
from vllm.tokenizers import TokenizerLike

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest

# Role label that Gemma4 emits at the start of the thinking channel.
# The model generates: <|channel>thought\n...reasoning...<channel|>
# This prefix must be stripped to expose only the actual reasoning content.
_THOUGHT_PREFIX = "thought\n"


class Gemma4ReasoningParser(BaseThinkingReasoningParser):
    """
    Reasoning parser for Google Gemma4 thinking models.

    Gemma4 uses <|channel>...<channel|> tokens to delimit reasoning/thinking
    content within its output. Thinking mode is activated by passing
    ``enable_thinking=True`` in the chat template kwargs, which injects a
    system turn containing <|think|> (token 98) to trigger chain-of-thought
    reasoning.

    Output pattern when thinking is enabled::

        <|channel>thought
        ...chain of thought reasoning...<channel|>
        Final answer text here.

    The ``thought\\n`` role label inside the channel delimiters is a
    structural artefact (analogous to ``user\\n`` in ``<|turn>user\\n...``).
    This parser strips it so that downstream consumers see only the
    actual reasoning text, consistent with the offline parser
    (``vllm.reasoning.gemma4_utils._strip_thought_label``).
    """

    # Signal to the serving layer that this parser can use token IDs
    # for non-streaming reasoning extraction.
    supports_token_ids = True

    def __init__(self, tokenizer: TokenizerLike, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        # Instance state for streaming prefix stripping.
        # Tracks only the reasoning text received from the base parser,
        # independent of current_text (which may contain pre-reasoning
        # content and lacks special token text due to
        # skip_special_tokens=True).
        self._reasoning_text: str = ""
        self._prefix_stripped: bool = False

    @property
    def start_token(self) -> str:
        """The token that starts reasoning content."""
        return "<|channel>"

    @property
    def end_token(self) -> str:
        """The token that ends reasoning content."""
        return "<channel|>"

    # ------------------------------------------------------------------
    # Non-streaming path
    # ------------------------------------------------------------------

    def extract_reasoning(
        self,
        model_output: str,
        request: "ChatCompletionRequest | ResponsesRequest",
        *,
        token_ids: Sequence[int] | None = None,
    ) -> tuple[str | None, str | None]:
        """Extract reasoning, stripping the ``thought\\n`` role label.

        When ``skip_special_tokens=True`` (the vLLM default), the
        ``<|channel>`` and ``<channel|>`` delimiters are stripped from
        ``model_output`` before this method is called, making text-based
        boundary detection impossible.  When *token_ids* is provided,
        this method falls back to token-ID-based extraction so that
        reasoning and content can still be separated correctly.
        """
        # --- Fast path: text-based matching (skip_special_tokens=False) ---
        if self.start_token in model_output or self.end_token in model_output:
            reasoning, content = super().extract_reasoning(model_output, request)
            if reasoning is not None:
                reasoning = _strip_thought_label(reasoning)
            return reasoning, content

        # --- Fallback: token-ID-based extraction (skip_special_tokens=True) ---
        if token_ids is not None:
            token_id_list = list(token_ids)
            start_pos: int | None = None
            end_pos: int | None = None
            for i, tid in enumerate(token_id_list):
                if tid == self.start_token_id and start_pos is None:
                    start_pos = i
                if tid == self.end_token_id:
                    end_pos = i
                    break  # use first end token

            if start_pos is not None and end_pos is not None:
                reasoning_ids = token_id_list[start_pos + 1:end_pos]
                content_ids = token_id_list[end_pos + 1:]
                reasoning = self.model_tokenizer.decode(
                    reasoning_ids, skip_special_tokens=True,
                )
                content = (
                    self.model_tokenizer.decode(
                        content_ids, skip_special_tokens=True,
                    )
                    if content_ids
                    else None
                )
                if reasoning:
                    reasoning = _strip_thought_label(reasoning)
                # Normalise empty strings to None for API consistency.
                return reasoning or None, content or None

        # No reasoning delimiters found at all.
        return None, model_output

    # ------------------------------------------------------------------
    # Streaming path
    # ------------------------------------------------------------------

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        """Extract streaming reasoning, stripping ``thought\\n`` from the
        first reasoning delta(s).

        The ``thought\\n`` prefix may arrive as a single delta or split
        across multiple deltas (e.g. ``"thought"`` then ``"\\n"``). We
        buffer early reasoning tokens until we can determine whether the
        prefix is present, then emit the buffered content minus the
        prefix.

        Unlike the previous implementation which reconstructed accumulated
        reasoning from ``current_text``, this uses instance state
        (``_reasoning_text``) to track only the reasoning content returned
        by the base parser. This is necessary because
        ``skip_special_tokens=True`` (the vLLM default) causes the
        ``<|channel>`` delimiter to be invisible in ``current_text``,
        making it impossible to separate pre-reasoning content from
        reasoning content via string matching.
        """
        result = super().extract_reasoning_streaming(
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
        )
        if result is None:
            return None

        if result.reasoning is None:
            return result

        # Accumulate ONLY the reasoning text from base parser results.
        # This is immune to pre-reasoning content pollution.
        self._reasoning_text += result.reasoning

        # Once the prefix has been handled, all subsequent reasoning
        # deltas pass through unchanged.
        if self._prefix_stripped:
            return result

        # ---- Prefix stripping logic ----

        # Case 1: We've accumulated enough to confirm the prefix is
        # present. Strip it and pass through the remainder.
        if self._reasoning_text.startswith(_THOUGHT_PREFIX):
            prefix_len = len(_THOUGHT_PREFIX)
            # How much reasoning was accumulated before this delta?
            prev_reasoning_len = len(self._reasoning_text) - len(result.reasoning)
            if prev_reasoning_len >= prefix_len:
                # Prefix was already consumed by prior deltas; this
                # delta is entirely real content — pass through.
                self._prefix_stripped = True
                return result
            else:
                # Part or all of the prefix is in this delta.
                chars_of_prefix_in_delta = prefix_len - prev_reasoning_len
                stripped = result.reasoning[chars_of_prefix_in_delta:]
                if stripped:
                    self._prefix_stripped = True
                    result.reasoning = stripped
                    return result
                else:
                    # This entire delta was prefix — suppress it.
                    # Don't set _prefix_stripped yet; there may be more
                    # prefix chars to consume in the next delta.
                    if len(self._reasoning_text) >= prefix_len:
                        self._prefix_stripped = True
                    return None

        # Case 2: Accumulated text is a strict prefix of
        # _THOUGHT_PREFIX (e.g. we've only seen "thou" so far).
        # Buffer by suppressing — we can't yet tell if this will
        # become the full prefix or diverge.
        if _THOUGHT_PREFIX.startswith(self._reasoning_text):
            return None

        # Case 3: Accumulated text doesn't match the thought prefix
        # at all. This means prior deltas were buffered (suppressed
        # by Case 2) but the text diverged. Re-emit the full
        # accumulated text to avoid data loss.
        self._prefix_stripped = True
        result.reasoning = self._reasoning_text
        return result


def _strip_thought_label(text: str) -> str:
    """Remove the ``thought\\n`` role label from the beginning of text.

    Mirrors ``vllm.reasoning.gemma4_utils._strip_thought_label`` from the
    offline parser.
    """
    if text.startswith(_THOUGHT_PREFIX):
        return text[len(_THOUGHT_PREFIX) :]
    return text
