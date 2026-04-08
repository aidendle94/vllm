# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for vllm.tool_parsers.gemma4_utils (offline tool call parsing)."""

from vllm.tool_parsers.gemma4_utils import (
    _parse_tool_arguments,
    has_tool_response_tag,
    parse_tool_calls,
)

# The Gemma4 escape token used as a string delimiter.
_ESCAPE = '<|"|>'


# ---------------------------------------------------------------------------
# _parse_tool_arguments
# ---------------------------------------------------------------------------


class TestParseToolArguments:
    """Unit tests for _parse_tool_arguments."""

    def test_empty_string(self):
        assert _parse_tool_arguments("") == {}

    def test_none_input(self):
        assert _parse_tool_arguments(None) == {}

    def test_whitespace_only(self):
        assert _parse_tool_arguments("   ") == {}

    def test_basic_string_value(self):
        args = f"city:{_ESCAPE}San Francisco{_ESCAPE}"
        result = _parse_tool_arguments(args)
        assert result == {"city": "San Francisco"}

    def test_multiple_string_values(self):
        args = f"city:{_ESCAPE}London{_ESCAPE},country:{_ESCAPE}UK{_ESCAPE}"
        result = _parse_tool_arguments(args)
        assert result == {"city": "London", "country": "UK"}

    def test_value_with_escape_token_delimiters(self):
        """The bug from issue #39069: delimiter-pair parsing is correct.

        In Gemma4 format, <|"|> is used as both opening and closing string
        delimiter. The parser should find the first pair and extract the
        value between them. With the old code, all <|"|> were replaced with
        " first, and then the regex ``[^"]*`` would stop at the first
        internal quote, truncating the value.
        """
        # description:<|"|>She said <|"|>  -> value is "She said "
        # The second <|"|> closes the string; the old code would have
        # produced just "She said " after regex fallback anyway, but the
        # important thing is the new parser does not crash or misbehave.
        args = f"description:{_ESCAPE}She said {_ESCAPE}"
        result = _parse_tool_arguments(args)
        assert result == {"description": "She said "}

    def test_value_with_literal_quotes_inside(self):
        """Value containing literal quote characters (not escape tokens)."""
        # If the model produces: key:<|"|>She said "hello"<|"|>
        # The raw args_str would have literal " inside the delimiters.
        args = f'key:{_ESCAPE}She said "hello"{_ESCAPE}'
        result = _parse_tool_arguments(args)
        assert result == {"key": 'She said "hello"'}

    def test_bare_numeric_value(self):
        args = "count:42"
        result = _parse_tool_arguments(args)
        assert result == {"count": "42"}

    def test_bare_boolean_value(self):
        args = "enabled:true"
        result = _parse_tool_arguments(args)
        assert result == {"enabled": "true"}

    def test_mixed_string_and_bare_values(self):
        args = f"name:{_ESCAPE}Alice{_ESCAPE},age:30"
        result = _parse_tool_arguments(args)
        assert result == {"name": "Alice", "age": "30"}

    def test_whitespace_around_values(self):
        args = f"key: {_ESCAPE}value{_ESCAPE}"
        result = _parse_tool_arguments(args)
        assert result == {"key": "value"}

    def test_unterminated_string_takes_rest(self):
        """If closing <|"|> is missing, take the rest of the string."""
        args = f"key:{_ESCAPE}no closing delimiter"
        result = _parse_tool_arguments(args)
        assert result == {"key": "no closing delimiter"}

    def test_multiple_pairs_with_spaces(self):
        args = f"first:{_ESCAPE}hello{_ESCAPE}, second:{_ESCAPE}world{_ESCAPE}"
        result = _parse_tool_arguments(args)
        assert result == {"first": "hello", "second": "world"}

    def test_empty_string_value(self):
        args = f"key:{_ESCAPE}{_ESCAPE}"
        result = _parse_tool_arguments(args)
        assert result == {"key": ""}

    def test_key_at_end_with_no_value(self):
        """Key followed by colon but nothing else -> empty string."""
        args = "key:"
        result = _parse_tool_arguments(args)
        assert result == {"key": ""}


# ---------------------------------------------------------------------------
# parse_tool_calls  (integration-level)
# ---------------------------------------------------------------------------


class TestParseToolCalls:
    """Tests for the top-level parse_tool_calls function."""

    def test_standard_format_single_call(self):
        text = (
            f"<|tool_call>call:get_weather{{city:{_ESCAPE}London{_ESCAPE}}}<tool_call|>"
        )
        calls = parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["name"] == "get_weather"
        assert calls[0]["arguments"] == {"city": "London"}

    def test_standard_format_with_turn_end(self):
        text = f"<|tool_call>call:search{{query:{_ESCAPE}vllm{_ESCAPE}}}<turn|>"
        calls = parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["name"] == "search"

    def test_no_tool_calls(self):
        text = "Just some regular text with no tool calls."
        calls = parse_tool_calls(text)
        assert calls == []

    def test_fallback_bare_call(self):
        text = f"call:greet{{name:{_ESCAPE}Bob{_ESCAPE}}}"
        calls = parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["name"] == "greet"
        assert calls[0]["arguments"] == {"name": "Bob"}

    def test_strict_mode_skips_fallback(self):
        text = f"call:greet{{name:{_ESCAPE}Bob{_ESCAPE}}}"
        calls = parse_tool_calls(text, strict=True)
        assert calls == []

    def test_value_with_internal_quotes_e2e(self):
        """End-to-end test: values with literal quotes inside delimiters."""
        text = (
            f"<|tool_call>call:describe{{"
            f'text:{_ESCAPE}He said "hi" to her{_ESCAPE}'
            f"}}<tool_call|>"
        )
        calls = parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["arguments"]["text"] == 'He said "hi" to her'


# ---------------------------------------------------------------------------
# has_tool_response_tag
# ---------------------------------------------------------------------------


class TestHasToolResponseTag:
    def test_present(self):
        assert has_tool_response_tag("some text<|tool_response>") is True

    def test_absent(self):
        assert has_tool_response_tag("some text<eos>") is False

    def test_trailing_whitespace(self):
        assert has_tool_response_tag("text<|tool_response>  ") is True
