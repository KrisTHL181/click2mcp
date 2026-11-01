"""Tests for signature-driven tool generation and argv synthesis."""

from __future__ import annotations

import inspect
import logging
import subprocess
from typing import Literal

import click
import pytest
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import ValidationError

from click2mcp.generator import generate_tools


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


def _tool(tools: list, name: str):
    """Return the tool called ``name``.

    :param tools: the generated tools.
    :param name: the wanted tool name.
    :returns: the tool.
    """
    return next(tool for tool in tools if tool.name == name)


@pytest.fixture
def rich_cli() -> click.Group:
    """A CLI with one command exercising every parameter shape.

    :returns: the root group.
    """

    @click.group()
    def cli() -> None:
        """Fixture root."""

    @cli.command()
    @click.argument("name")
    @click.argument("files", nargs=-1)
    @click.option("--token", required=True, help="Authentication token to send with the request")
    @click.option("--long-name")
    @click.option("--color", type=click.Choice(["red", "green"]), default="red")
    @click.option("--count", type=click.IntRange(1, 10), default=1)
    @click.option("--flag", is_flag=True)
    @click.option("--tag", multiple=True)
    @click.option("--maybe", default=None)
    def all_types(name, files, token, long_name, color, count, flag, tag, maybe) -> None:
        """Do a thing."""

    return cli


def test_tool_name_description_and_signature(rich_cli: click.Group) -> None:
    """The name, description, types, and required-first signature are correct."""
    recorder = Recorder()
    tools = generate_tools(cli=rich_cli, invoke=recorder, app="mycli")

    assert [tool.name for tool in tools] == ["mycli_all_types"]
    tool = tools[0]
    assert tool.description == "Do a thing.\n\nCLI: mycli all-types [ARGS] [OPTIONS]"

    signature = inspect.signature(tool.fn)
    names = list(signature.parameters)
    # Python forbids a defaulted parameter before a bare one, so required params
    # must lead even though click declared the options after the arguments.
    assert names[:2] == ["name", "token"]
    assert names.index("token") < names.index("long_name")
    assert "long_name" in names
    assert signature.parameters["long_name"].annotation == (str | None)
    assert signature.parameters["color"].annotation == Literal["red", "green"]
    assert signature.parameters["count"].annotation is int
    assert signature.parameters["flag"].annotation is bool
    assert signature.parameters["tag"].annotation == (list[str] | None)
    assert signature.parameters["files"].annotation == (list[str] | None)

    properties = tool.parameters["properties"]
    assert properties["color"]["enum"] == ["red", "green"]
    assert properties["count"]["type"] == "integer"
    assert properties["flag"]["type"] == "boolean"
    assert properties["tag"]["anyOf"][0]["type"] == "array"
    assert properties["files"]["anyOf"][0]["type"] == "array"
    assert set(tool.parameters["required"]) == {"name", "token"}
    # Required options keep the CLI's own help; title noise is stripped.
    assert properties["token"]["description"] == "Authentication token to send with the request"
    assert "title" not in tool.parameters


def test_defaults_and_unset_sentinel(rich_cli: click.Group) -> None:
    """Click defaults appear, None becomes nullable, and the sentinel never leaks."""
    recorder = Recorder()
    tool = _tool(generate_tools(cli=rich_cli, invoke=recorder, app="mycli"), "mycli_all_types")
    signature = inspect.signature(tool.fn)
    properties = tool.parameters["properties"]

    assert signature.parameters["color"].default == "red"
    assert properties["color"]["default"] == "red"
    assert signature.parameters["count"].default == 1
    assert properties["count"]["default"] == 1
    assert signature.parameters["flag"].default is False
    assert signature.parameters["maybe"].default is None
    assert signature.parameters["tag"].default is None
    assert properties["maybe"]["default"] is None
    assert properties["tag"]["default"] is None
    # The sentinel used for an option with no declared default must not appear.
    for param in signature.parameters.values():
        assert "Sentinel" not in repr(param.default)
        assert "UNSET" not in repr(param.default)


def test_argv_rendering_through_tool_call(rich_cli: click.Group) -> None:
    """Options precede ``--``, values are one token, repeats expand, defaults drop."""
    recorder = Recorder(stdout="result")
    tool = _tool(generate_tools(cli=rich_cli, invoke=recorder, app="mycli"), "mycli_all_types")

    assert (
        tool.fn(
            name="n",
            token="t",
            long_name="x",
            color="green",
            count=3,
            flag=True,
            tag=["a", "b"],
            files=["f1", "f2"],
        )
        == "result"
    )

    assert recorder.calls[-1] == [
        "all-types",
        "--token=t",
        "--long-name=x",
        "--color=green",
        "--count=3",
        "--flag",
        "--tag=a",
        "--tag=b",
        "--",
        "n",
        "f1",
        "f2",
    ]
    assert recorder.calls[-1].index("--") < recorder.calls[-1].index("n")

    # None and default values are omitted; nothing but the path is left.
    recorder.calls.clear()
    tool.fn(name="n", token="t", color="red", flag=False, maybe=None)
    assert recorder.calls[-1] == ["all-types", "--token=t", "--", "n"]


def test_dual_flag_negative_spelling() -> None:
    """A dual flag uses its negative spelling when the value is False."""

    @click.group()
    def cli() -> None:
        """Root."""

    @cli.command()
    @click.option("--cache/--no-cache", default=True)
    def run(cache) -> None:
        """Run."""

    recorder = Recorder()
    tool = _tool(generate_tools(cli=cli, invoke=recorder, app="mycli"), "mycli_run")

    tool.fn(cache=False)
    assert recorder.calls[-1] == ["run", "--no-cache"]
    tool.fn(cache=True)
    assert recorder.calls[-1] == ["run"]


def test_name_collision_skips_only_that_command(caplog: pytest.LogCaptureFixture) -> None:
    """A parameter name collision drops one command; the rest survive."""

    @click.group()
    def cli() -> None:
        """Root."""

    @cli.command()
    @click.argument("json")
    @click.option("--json", "json")
    def bad(json) -> None:
        """Bad."""

    @cli.command()
    def good() -> None:
        """Good."""

    with caplog.at_level(logging.WARNING, logger="click2mcp.generator"):
        tools = generate_tools(cli=cli, invoke=Recorder(), app="mycli")

    assert [tool.name for tool in tools] == ["mycli_good"]
    assert any("skipping" in record.message for record in caplog.records)


def test_exclude_and_include(caplog: pytest.LogCaptureFixture) -> None:
    """Exclude prunes subtrees; include selects them; a stray include only logs."""

    @click.group()
    def cli() -> None:
        """Root."""

    @cli.group()
    def admin() -> None:
        """Admin."""

    @admin.command()
    def secret() -> None:
        """Secret."""

    @cli.command()
    def public() -> None:
        """Public."""

    recorder = Recorder()
    excluded = generate_tools(cli=cli, invoke=recorder, app="mycli", exclude=frozenset({("admin",)}))
    assert [tool.name for tool in excluded] == ["mycli_public"]

    included = generate_tools(cli=cli, invoke=recorder, app="mycli", include=(("admin",),))
    assert [tool.name for tool in included] == ["mycli_admin_secret"]

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="click2mcp.generator"):
        empty = generate_tools(cli=cli, invoke=recorder, app="mycli", include=(("nope",),))
    assert empty == []
    assert any("matches no command" in record.message for record in caplog.records)


def test_none_cli_returns_no_tools() -> None:
    """A CLI that could not be loaded costs the typed tools, not the call."""
    assert generate_tools(cli=None, invoke=Recorder(), app="mycli") == []


@pytest.fixture
def nested_cli() -> click.Group:
    """A root group with a flag and a version option, and a nested group.

    :returns: the root group.
    """

    @click.group()
    @click.option("--verbose", is_flag=True)
    @click.version_option("1.0")
    def cli(verbose) -> None:
        """Root."""

    @cli.group()
    @click.option("--profile")
    def sub(profile) -> None:
        """Sub."""

    @sub.command()
    @click.argument("n")
    def run(n) -> None:
        """Run."""

    @sub.command()
    @click.option("--verbose", is_flag=True)
    @click.argument("n")
    def shadow(n, verbose) -> None:
        """Shadow."""

    return cli


def test_inherited_group_options(nested_cli: click.Group) -> None:
    """Ancestor options are exposed and emitted at their command path level."""
    recorder = Recorder()
    tools = generate_tools(cli=nested_cli, invoke=recorder, app="mycli")
    run_tool = _tool(tools, "mycli_sub_run")
    signature = inspect.signature(run_tool.fn)

    assert "profile" in signature.parameters
    assert "verbose" in signature.parameters
    assert "version" not in signature.parameters

    assert run_tool.fn(verbose=True, profile="x", n="n") == "ok"
    assert recorder.calls[-1] == ["--verbose", "sub", "--profile=x", "run", "--", "n"]

    # A leaf option with the same kwarg wins over the inherited root option, so the
    # flag is emitted after the sub-command, not before it.
    shadow_tool = _tool(tools, "mycli_sub_shadow")
    shadow_tool.fn(verbose=True, n="n")
    assert recorder.calls[-1] == ["sub", "shadow", "--verbose", "--", "n"]


def test_inherit_group_options_can_be_disabled(nested_cli: click.Group) -> None:
    """``inherit_group_options=False`` exposes only the leaf's own parameters."""
    tools = generate_tools(cli=nested_cli, invoke=Recorder(), app="mycli", inherit_group_options=False)
    signature = inspect.signature(_tool(tools, "mycli_sub_run").fn)
    assert list(signature.parameters) == ["n"]


def test_runner_routes_to_invoke_and_raises_tool_error(rich_cli: click.Group) -> None:
    """The generated body calls the injected runner and surfaces failures as ToolError."""
    recorder = Recorder(returncode=2, stdout="", stderr="boom")
    tool = _tool(generate_tools(cli=rich_cli, invoke=recorder, app="mycli"), "mycli_all_types")

    with pytest.raises(ToolError) as excinfo:
        tool.fn(name="n", token="t")

    assert recorder.calls[-1] == ["all-types", "--token=t", "--", "n"]
    assert "exited with code 2" in str(excinfo.value)
    assert "boom" in str(excinfo.value)


def test_prompted_options_are_required() -> None:
    """A prompt can never be answered over MCP, so prompted options must be required."""

    @click.group()
    def cli() -> None:
        """Root."""

    @cli.command()
    @click.option("--phone", prompt="Phone")
    @click.password_option("--password", prompt="Password")
    @click.option("--nickname")
    def login(phone, password, nickname) -> None:
        """Log in."""

    recorder = Recorder()
    tool = _tool(generate_tools(cli=cli, invoke=recorder, app="mycli"), "mycli_login")

    assert set(tool.parameters["required"]) == {"phone", "password"}
    # Required params must not be widened to ``string | null``.
    assert tool.parameters["properties"]["phone"] == {"type": "string"}
    assert tool.parameters["properties"]["password"] == {"type": "string"}
    # An ordinary optional option stays optional and nullable.
    assert "nickname" not in tool.parameters["required"]
    assert "anyOf" in tool.parameters["properties"]["nickname"]

    with pytest.raises(ValidationError):
        tool.fn_metadata.validate_arguments({"nickname": "nick"})

    tool.fn_metadata.validate_arguments({"phone": "123", "password": "secret"})
    tool.fn(phone="123", password="secret")
    assert recorder.calls[-1] == ["login", "--phone=123", "--password=secret"]


def test_tool_prefix_defaults_to_app(rich_cli: click.Group) -> None:
    """``tool_prefix=None`` keeps the historical ``{app}_`` name."""
    tools = generate_tools(cli=rich_cli, invoke=Recorder(), app="mycli", tool_prefix=None)
    assert [tool.name for tool in tools] == ["mycli_all_types"]


def test_empty_tool_prefix_drops_app(rich_cli: click.Group) -> None:
    """``tool_prefix=""`` names the tool after the command path alone."""
    tools = generate_tools(cli=rich_cli, invoke=Recorder(), app="mycli", tool_prefix="")
    assert [tool.name for tool in tools] == ["all_types"]


def test_custom_tool_prefix_is_normalized(rich_cli: click.Group) -> None:
    """A custom prefix gets a trailing ``_``; spaces and hyphens become ``_``."""
    recorder = Recorder()
    assert [tool.name for tool in generate_tools(cli=rich_cli, invoke=recorder, app="mycli", tool_prefix="auto")] == [
        "auto_all_types"
    ]
    assert [
        tool.name for tool in generate_tools(cli=rich_cli, invoke=recorder, app="mycli", tool_prefix="au-to")
    ] == ["au_to_all_types"]
    assert [
        tool.name for tool in generate_tools(cli=rich_cli, invoke=recorder, app="mycli", tool_prefix="my cli")
    ] == ["my_cli_all_types"]


def test_invalid_tool_prefix_raises(rich_cli: click.Group) -> None:
    """A prefix that cannot be part of an identifier is refused, naming the value."""
    with pytest.raises(ValueError, match="2fa"):
        generate_tools(cli=rich_cli, invoke=Recorder(), app="mycli", tool_prefix="2fa")


def test_empty_prefix_collision_skips_only_that_command(caplog: pytest.LogCaptureFixture) -> None:
    """Two paths normalizing to one name still collide when the prefix is empty."""

    @click.group()
    def cli() -> None:
        """Root."""

    @cli.command(name="a-b")
    def dash() -> None:
        """Dash."""

    @cli.command(name="a_b")
    def underscore() -> None:
        """Underscore."""

    with caplog.at_level(logging.WARNING, logger="click2mcp.generator"):
        tools = generate_tools(cli=cli, invoke=Recorder(), app="mycli", tool_prefix="")

    assert [tool.name for tool in tools] == ["a_b"]
    assert any("already used by" in record.message for record in caplog.records)
