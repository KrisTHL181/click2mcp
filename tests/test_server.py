"""Tests for the ClickMCP wrapper and its escape hatches."""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys

import click
import pytest
from mcp.server.mcpserver.exceptions import ToolError

from click2mcp.server import ClickMCP, _excluded_path, _group_prefixes


class Recorder:
    """A fake ``invoke`` that records argv and returns a canned process result.

    :param returncode: exit status to report.
    :param stdout: stdout to report.
    :param stderr: stderr to report.
    """

    def __init__(self, returncode: int = 0, stdout: str = "ok", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.calls: list[list[str]] = []

    def __call__(self, tokens: list[str]) -> subprocess.CompletedProcess[str]:
        """Record ``tokens`` and return the canned result.

        :param tokens: the CLI arguments.
        :returns: a completed process.
        """
        self.calls.append(list(tokens))
        return subprocess.CompletedProcess(tokens, self.returncode, self.stdout, self.stderr)


class FakeServer:
    """A stand-in for ``MCPServer`` that only records ``run`` calls."""

    def __init__(self) -> None:
        self.transports: list[str] = []

    def run(self, transport: str = "stdio") -> None:
        """Record the transport.

        :param transport: the transport name.
        """
        self.transports.append(transport)


def _tools(server) -> dict[str, object]:
    """Map tool names to tools for a built server.

    :param server: the ``MCPServer``.
    :returns: tool name -> tool.
    """
    return {tool.name: tool for tool in server._tool_manager.list_tools()}


@pytest.fixture
def cli() -> click.Group:
    """A small CLI with a plain command and a nested group.

    :returns: the root group.
    """

    @click.group()
    def cli() -> None:
        """Fixture root."""

    @cli.command()
    @click.argument("name")
    def greet(name) -> None:
        """Greet someone."""

    @cli.group()
    def admin() -> None:
        """Admin commands."""

    @admin.command()
    def secret() -> None:
        """Secret command."""

    return cli


def test_escape_hatches_toggle(cli: click.Group) -> None:
    """Both escape hatches are present by default and absent when disabled."""
    enabled = ClickMCP(app="mycli", cli=cli, invoke=Recorder())
    assert set(_tools(enabled.server)) == {"mycli_admin_secret", "mycli_greet", "mycli_help", "mycli_run"}
    assert "mycli_run" in enabled.server.instructions

    disabled = ClickMCP(app="mycli", cli=cli, invoke=Recorder(), escape_hatches=False)
    assert set(_tools(disabled.server)) == {"mycli_admin_secret", "mycli_greet"}
    assert "mycli_run" not in disabled.server.instructions
    assert "mycli_help" not in disabled.server.instructions


def test_run_refuses_excluded_paths_and_empty_commands(cli: click.Group) -> None:
    """A blocked path never reaches ``invoke``; an empty command is rejected."""
    recorder = Recorder()
    bridge = ClickMCP(
        app="mycli",
        cli=cli,
        exclude=frozenset({("admin", "secret")}),
        invoke=recorder,
    )
    tools = _tools(bridge.server)
    # The excluded leaf is absent from the typed tools, too.
    assert "mycli_admin_secret" not in tools

    run_tool = tools["mycli_run"]
    with pytest.raises(ToolError, match="full-screen"):
        run_tool.fn(args="admin secret")
    assert recorder.calls == []

    with pytest.raises(ToolError, match="No command given"):
        run_tool.fn(args="   ")
    assert recorder.calls == []


def test_excluded_path_matching_semantics() -> None:
    """A trailing option still matches; a leading option does not."""
    exclude = frozenset({("admin", "secret")})
    assert _excluded_path(["admin", "secret", "--force"], exclude) == ("admin", "secret")
    assert _excluded_path(["--profile", "x", "admin", "secret"], exclude) is None
    assert _excluded_path(["admin"], exclude) is None


def test_explicit_launcher_is_used_verbatim(monkeypatch: pytest.MonkeyPatch) -> None:
    """A given launcher replaces executable discovery and reaches subprocess.run."""
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(argv, **kwargs):
        calls.append((list(argv), kwargs))
        return subprocess.CompletedProcess(argv, 0, "out", "")

    monkeypatch.setattr("click2mcp.server.subprocess.run", fake_run)
    bridge = ClickMCP(
        app="mycli",
        cli=None,
        launcher=["custom-cli", "--flag"],
        env={"NO_COLOR": "0", "MY_VAR": "1"},
    )
    _tools(bridge.server)["mycli_help"].fn(path="admin")

    assert calls[0][0] == ["custom-cli", "--flag", "admin", "--help"]
    assert calls[0][1]["stdin"] == subprocess.DEVNULL
    assert calls[0][1]["capture_output"] is True
    assert calls[0][1]["text"] is True
    env = calls[0][1]["env"]
    assert env["NO_COLOR"] == "0"
    assert env["TERM"] == "dumb"
    assert env["PYTHONIOENCODING"] == "utf-8"
    assert env["MY_VAR"] == "1"


def test_module_is_the_launcher_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no installed executable, ``python -m module`` is used."""
    monkeypatch.setattr("click2mcp.server.shutil.which", lambda app: None)
    calls: list[list[str]] = []

    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, "out", "")

    monkeypatch.setattr("click2mcp.server.subprocess.run", fake_run)
    bridge = ClickMCP(app="click2mcpnotinstalled", cli=None, module="mycli.main")
    _tools(bridge.server)["click2mcpnotinstalled_help"].fn()

    assert calls[-1] == [sys.executable, "-m", "mycli.main", "--help"]


def test_missing_launcher_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neither an executable nor a module is a clear construction-time failure."""
    monkeypatch.setattr("click2mcp.server.shutil.which", lambda app: None)
    with pytest.raises(RuntimeError, match="no module was given"):
        ClickMCP(app="click2mcpnotinstalled", cli=None)


def test_group_prefix_parser() -> None:
    """Dotted and spaced paths parse; empty entries are refused."""
    assert _group_prefixes("browse,tools.creator") == (("browse",), ("tools", "creator"))
    assert _group_prefixes("tools creator") == (("tools", "creator"),)
    assert _group_prefixes(" a . b ") == (("a", "b"),)
    with pytest.raises(argparse.ArgumentTypeError, match="empty command path"):
        _group_prefixes("a,,b")
    with pytest.raises(argparse.ArgumentTypeError, match="empty command path"):
        _group_prefixes("")


def test_run_parses_groups(cli: click.Group) -> None:
    """``--groups`` rebuilds the server; no argument keeps the eager one."""
    bridge = ClickMCP(app="mycli", cli=cli, invoke=Recorder())

    built: list[object] = []
    narrowed = FakeServer()

    def fake_build(include=None):
        built.append(include)
        return narrowed

    bridge.build_server = fake_build
    bridge.run(["--groups", "admin.secret"])
    assert built == [(("admin", "secret"),)]
    assert narrowed.transports == ["stdio"]

    default = FakeServer()
    bridge.server = default
    bridge.run([])
    assert default.transports == ["stdio"]


def test_build_server_include_narrows_tools(cli: click.Group) -> None:
    """``build_server(include=...)`` overrides the constructor's include."""
    bridge = ClickMCP(app="mycli", cli=cli, invoke=Recorder())
    assert set(_tools(bridge.build_server())) == {
        "mycli_admin_secret",
        "mycli_greet",
        "mycli_help",
        "mycli_run",
    }

    narrowed = bridge.build_server(include=(("admin",),))
    assert set(_tools(narrowed)) == {"mycli_admin_secret", "mycli_help", "mycli_run"}

    constructor_narrowed = ClickMCP(app="mycli", cli=cli, include=(("greet",),), invoke=Recorder())
    assert set(_tools(constructor_narrowed.server)) == {"mycli_greet", "mycli_help", "mycli_run"}


def test_custom_invoke_is_shared_by_tools_and_hatches(cli: click.Group) -> None:
    """The injected runner serves generated tools, help, and run alike."""
    recorder = Recorder(stdout="result")
    bridge = ClickMCP(app="mycli", cli=cli, invoke=recorder)
    tools = _tools(bridge.server)

    assert tools["mycli_greet"].fn(name="n") == "result"
    assert tools["mycli_run"].fn(args="greet n") == "result"
    assert tools["mycli_help"].fn(path="admin") == "result"

    assert recorder.calls == [["greet", "--", "n"], ["greet", "n"], ["admin", "--help"]]


def test_escape_hatch_errors_are_tool_errors(cli: click.Group) -> None:
    """Non-zero exits from either entry point reach the client as ToolError."""
    recorder = Recorder(returncode=1, stdout="", stderr="bad\n")
    bridge = ClickMCP(app="mycli", cli=cli, invoke=recorder)
    tools = _tools(bridge.server)

    with pytest.raises(ToolError, match="exited with code 1"):
        tools["mycli_help"].fn(path="admin")
    with pytest.raises(ToolError, match="exited with code 1"):
        tools["mycli_run"].fn(args="admin secret")


def test_callable_cli_is_resolved_once(cli: click.Group) -> None:
    """A loader is called exactly once during construction."""
    calls: list[int] = []

    def load() -> click.Group:
        calls.append(1)
        return cli

    bridge = ClickMCP(app="mycli", cli=load, invoke=Recorder())
    assert calls == [1]
    assert "mycli_greet" in _tools(bridge.server)


def test_failed_cli_loader_serves_escape_hatches(caplog: pytest.LogCaptureFixture) -> None:
    """A loader that raises costs the typed tools, not the server."""

    def load() -> click.Group:
        raise RuntimeError("extension blew up")

    with caplog.at_level(logging.WARNING, logger="click2mcp.server"):
        bridge = ClickMCP(app="mycli", cli=load, invoke=Recorder())

    assert set(_tools(bridge.server)) == {"mycli_help", "mycli_run"}
    assert any("could not load the CLI" in record.message for record in caplog.records)


def test_empty_tool_prefix_keeps_escape_hatches_prefixed(cli: click.Group) -> None:
    """``tool_prefix=""`` drops the prefix from typed tools but not from help/run."""
    bridge = ClickMCP(app="mycli", cli=cli, invoke=Recorder(), tool_prefix="")
    assert set(_tools(bridge.server)) == {"admin_secret", "greet", "mycli_help", "mycli_run"}


def test_instructions_use_the_resolved_prefix(cli: click.Group) -> None:
    """The default instructions name tools the server actually exposes."""
    unprefixed = ClickMCP(app="mycli", cli=cli, invoke=Recorder(), tool_prefix="")
    assert "`browse_following_users`" in unprefixed.server.instructions
    assert "mycli_browse_following_users" not in unprefixed.server.instructions
    # The escape hatches keep their app prefix even with no typed-tool prefix.
    assert "`mycli_help(path)`" in unprefixed.server.instructions
    assert "`mycli_run(args)`" in unprefixed.server.instructions

    default = ClickMCP(app="mycli", cli=cli, invoke=Recorder())
    assert "`mycli_browse_following_users`" in default.server.instructions

    custom = ClickMCP(app="mycli", cli=cli, invoke=Recorder(), tool_prefix="auto")
    assert "`auto_browse_following_users`" in custom.server.instructions


def test_invalid_tool_prefix_raises_from_constructor(cli: click.Group) -> None:
    """A bad prefix is a construction-time ``ValueError``, not a silent fallback."""
    with pytest.raises(ValueError, match="2fa"):
        ClickMCP(app="mycli", cli=cli, invoke=Recorder(), tool_prefix="2fa")
