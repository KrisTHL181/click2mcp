"""Tests for JSON output shaping."""

from __future__ import annotations

import subprocess

import click
import pytest

from click2mcp._output import compact_json, shape_output
from click2mcp.generator import generate_tools
from click2mcp.server import ClickMCP


class Recorder:
    """A fake ``invoke`` that records argv and returns a canned process result.

    :param stdout: stdout to report.
    :param stderr: stderr to report.
    """

    def __init__(self, stdout: str = "ok", stderr: str = "") -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.calls: list[list[str]] = []

    def __call__(self, tokens: list[str]) -> subprocess.CompletedProcess[str]:
        """Record ``tokens`` and return the canned result.

        :param tokens: the CLI arguments.
        :returns: a completed process.
        """
        self.calls.append(list(tokens))
        return subprocess.CompletedProcess(tokens, 0, self.stdout, self.stderr)


def test_pretty_json_is_compacted() -> None:
    """Indentation and separator whitespace are dropped; key order survives."""
    pretty = '{\n  "b": 2,\n  "a": [1, 2, {"c": "d"}]\n}\n'
    assert compact_json(pretty) == '{"b":2,"a":[1,2,{"c":"d"}]}'


def test_unicode_is_not_escaped() -> None:
    """Non-ASCII text stays readable instead of becoming ``\\u`` escapes."""
    assert compact_json('{\n  "name": "中文"\n}') == '{"name":"中文"}'


def test_non_json_is_untouched() -> None:
    """Help text, JSON Lines, and a banner around JSON pass through unchanged."""
    help_text = "Usage: mycli [OPTIONS]\n\n  Do a thing.\n"
    json_lines = '{"a": 1}\n{"b": 2}\n'
    banner = 'loading...\n{"a": 1}\n'
    assert compact_json(help_text) == help_text
    assert compact_json(json_lines) == json_lines
    assert compact_json(banner) == banner


def test_empty_output_is_untouched() -> None:
    """Whitespace-only output is returned as-is."""
    assert compact_json("") == ""
    assert compact_json("  \n") == "  \n"


def test_already_compact_json_is_untouched() -> None:
    """No rewrite when re-serialization would not shorten the text."""
    assert compact_json('{"a":1}') == '{"a":1}'
    # A bare number normalizes to a longer spelling, so the original wins.
    assert compact_json("1e5") == "1e5"


def test_object_is_returned_as_data() -> None:
    """A JSON object comes back parsed, so the client gets a real object."""
    assert shape_output('{\n  "answer": 42\n}\n') == {"answer": 42}


def test_array_is_returned_as_data() -> None:
    """A JSON array comes back parsed too."""
    assert shape_output('[\n  {"id": 1},\n  {"id": 2}\n]\n') == [{"id": 1}, {"id": 2}]


def test_scalar_json_stays_text() -> None:
    """A bare scalar is not a document worth handing back as data."""
    assert shape_output("1e5") == "1e5"
    assert shape_output('"hello"') == '"hello"'


def test_non_json_stays_text() -> None:
    """Anything that is not one complete JSON document reaches the client as-is."""
    assert shape_output("Usage: mycli [OPTIONS]\n") == "Usage: mycli [OPTIONS]\n"
    assert shape_output("") == ""
    assert shape_output("  \n") == "  \n"


@pytest.fixture
def cli() -> click.Group:
    """A CLI with one command.

    :returns: the root group.
    """

    @click.group()
    def cli() -> None:
        """Root."""

    @cli.command()
    def show() -> None:
        """Show something."""

    return cli


def test_generated_tools_return_json_data(cli: click.Group) -> None:
    """A typed tool hands its CLI's JSON object back as an object."""
    recorder = Recorder(stdout='{\n  "answer": 42\n}\n')
    tool = generate_tools(cli=cli, invoke=recorder, app="mycli")[0]

    assert tool.fn() == {"answer": 42}


def test_structured_output_carries_the_object(cli: click.Group) -> None:
    """The wire result holds the object itself, not a JSON-encoded string."""
    recorder = Recorder(stdout='{\n  "answer": 42\n}\n')
    tool = generate_tools(cli=cli, invoke=recorder, app="mycli")[0]

    result = tool.fn_metadata.convert_result(tool.fn())

    assert result.structured_content == {"result": {"answer": 42}}


def test_run_escape_hatch_returns_json_data(cli: click.Group) -> None:
    """``{app}_run`` hands the object back the same way as the typed tools."""
    recorder = Recorder(stdout='{\n  "answer": 42\n}\n')
    bridge = ClickMCP(app="mycli", cli=cli, invoke=recorder)
    run_tool = next(tool for tool in bridge.server._tool_manager.list_tools() if tool.name == "mycli_run")

    assert run_tool.fn(args="show") == {"answer": 42}


def test_stderr_fallback_is_data_too(cli: click.Group) -> None:
    """When stdout is empty, a JSON warning on stderr is returned as an object."""
    recorder = Recorder(stdout="", stderr='{\n  "warning": "low disk"\n}\n')
    tool = generate_tools(cli=cli, invoke=recorder, app="mycli")[0]

    assert tool.fn() == {"warning": "low disk"}
