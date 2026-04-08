# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 Google Inc. HuggingFace Inc. team. All rights reserved.

"""Gemma4 tool call parsing utilities for offline inference.

Standalone functions that parse decoded model text to extract tool calls
from Gemma4 models. These are pure-Python utilities with zero heavy
dependencies — they work on raw decoded strings from any inference
backend (vLLM, HuggingFace, TGI, etc.).

For the OpenAI-compatible API server tool parser (streaming +
non-streaming), see ``vllm.tool_parsers.gemma4_tool_parser``.
For thinking/reasoning output parsing, see
``vllm.reasoning.gemma4_utils``.

Usage with vLLM offline inference::

    from vllm import LLM, SamplingParams
    from vllm.tool_parsers.gemma4_utils import (
        parse_tool_calls,
        has_tool_response_tag,
    )

    llm = LLM(model="google/gemma-4-it")
    outputs = llm.generate(prompt, SamplingParams(...))
    text = tokenizer.decode(outputs[0].outputs[0].token_ids, skip_special_tokens=False)

    # Extract tool calls
    tool_calls = parse_tool_calls(text)
    for tc in tool_calls:
        print(f"{tc['name']}({tc['arguments']})")

Ported from ``transformers.models.gemma4.utils_gemma4`` so that vLLM users
do not need a transformers dependency for output parsing.
"""

import regex as re

# Tool call delimiter tokens as they appear in decoded text.
# Standard format: <|tool_call>call:name{args}<tool_call|>
_TOOL_CALL_START_TAG = "<|tool_call>"
_TOOL_CALL_END_TAG = "<tool_call|>"
_TOOL_RESPONSE_START_TAG = "<|tool_response>"

# Gemma4 escape token as it appears in decoded text.
_ESCAPE_TOKEN = '<|"|>'


def _parse_tool_arguments(args_str: str) -> dict[str, str]:
    """Parse tool call arguments from the Gemma4 compact format.

    Handles the ``key:<|"|>value<|"|>`` format used by Gemma4.
    Parses <|"|> delimiters directly to correctly handle values
    containing internal quotes or other special characters.

    Args:
        args_str: Raw argument string from inside ``call:name{...}``.

    Returns:
        Dictionary of argument name → value (all values as strings).
    """
    if not args_str or not args_str.strip():
        return {}

    result: dict[str, str] = {}
    i = 0
    n = len(args_str)

    while i < n:
        # Skip whitespace and commas
        while i < n and args_str[i] in (" ", ",", "\n", "\t"):
            i += 1
        if i >= n:
            break

        # Parse key (unquoted, ends at ':')
        key_start = i
        while i < n and args_str[i] != ":":
            i += 1
        if i >= n:
            break
        key = args_str[key_start:i].strip()
        i += 1  # skip ':'

        # Skip whitespace after ':'
        while i < n and args_str[i] in (" ", "\n", "\t"):
            i += 1
        if i >= n:
            result[key] = ""
            break

        # String value delimited by <|"|>
        if args_str[i:].startswith(_ESCAPE_TOKEN):
            i += len(_ESCAPE_TOKEN)
            end_pos = args_str.find(_ESCAPE_TOKEN, i)
            if end_pos == -1:
                # Unterminated string — take rest
                result[key] = args_str[i:]
                break
            result[key] = args_str[i:end_pos]
            i = end_pos + len(_ESCAPE_TOKEN)

        # Bare value (number, boolean, unquoted string)
        else:
            val_start = i
            while i < n and args_str[i] not in (",", "}"):
                i += 1
            value = args_str[val_start:i].strip()
            # Convert to string representation for consistency
            result[key] = value

    return result


def parse_tool_calls(text: str, *, strict: bool = False) -> list[dict]:
    """Parse tool calls from decoded Gemma4 model output.

    Uses a tiered parsing strategy to handle known output variations in
    Gemma4 models, which may emit
    non-standard tool call formats.

    Parsing tiers:
        1. **Standard**: ``<|tool_call>call:name{args}<tool_call|>``
           (special token IDs 48/49 in decoded text)
        2. **Fallback** (when ``strict=False``): bare ``call:name{args}``
           patterns, including ``<call>name{args}`` (fragmented tokens from
           multimodal inputs)

    Args:
        text: Decoded model output text (from ``tokenizer.decode(...,
            skip_special_tokens=False)``).
        strict: If ``True``, only match the standard ``<|tool_call>`` format.
            If ``False`` (default), also try fallback patterns for
            known Gemma4 output variations.

    Returns:
        A list of dicts, each with keys:
            - ``"name"``: The tool function name (e.g. ``"get_weather"``).
            - ``"arguments"``: A dict of argument name → value.

    Example::

        >>> from vllm.tool_parsers.gemma4_utils import parse_tool_calls
        >>> output = tokenizer.decode(outputs[0], skip_special_tokens=False)
        >>> tool_calls = parse_tool_calls(output)
        >>> for tc in tool_calls:
        ...     print(f"Call: {tc['name']}({tc['arguments']})")
    """
    results = []

    # Tier 1: Standard format with special tokens.
    # <|tool_call>call:name{args}<tool_call|>
    # Note: Some Gemma4 models emit <turn|> instead of <tool_call|>.
    standard_pattern = r"<\|tool_call\>call:(\w+)\{(.*?)\}(?:<tool_call\|>|<turn\|>)"
    for match in re.finditer(standard_pattern, text, re.DOTALL):
        name, args_str = match.group(1), match.group(2)
        results.append(
            {
                "name": name,
                "arguments": _parse_tool_arguments(args_str),
            }
        )

    if results or strict:
        return results

    # Tier 2: Fallback for known Gemma4 output variations.
    # Matches: <call>name{args}, call:name{args}, or bare call:name{args}<eos>
    fallback_pattern = r"(?:<call>|(?:^|\s)call:)(\w+)\{(.*?)\}"
    for match in re.finditer(fallback_pattern, text, re.DOTALL):
        name, args_str = match.group(1), match.group(2)
        results.append(
            {
                "name": name,
                "arguments": _parse_tool_arguments(args_str),
            }
        )

    return results


def has_tool_response_tag(text: str) -> bool:
    """Check if model output properly ends with a tool response tag.

    Some Gemma4 models sometimes emit ``<eos>`` instead of
    ``<|tool_response>`` after a tool call. This helper detects
    whether the model used the proper termination, so callers can
    decide whether to inject ``<|tool_response>`` into the next prompt.

    Args:
        text: Decoded model output text.

    Returns:
        ``True`` if the output ends with ``<|tool_response>``
        (proper behavior), ``False`` otherwise.

    Example::

        >>> from vllm.tool_parsers.gemma4_utils import has_tool_response_tag
        >>> if not has_tool_response_tag(model_output):
        ...     # Model used <eos> instead — inject <|tool_response> manually
        ...     next_prompt = "<|tool_response>" + tool_result
    """
    stripped = text.rstrip()
    return stripped.endswith(_TOOL_RESPONSE_START_TAG)
