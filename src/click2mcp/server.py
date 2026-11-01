"""MCP server bridging a Click CLI to AI agents.

Every leaf command of the CLI is exposed as its own typed MCP tool, generated from
the Click command tree at startup by :mod:`click2mcp.generator`. The model sees real
parameter names, types, defaults, and enums rather than having to spell a command
line from memory. A consumer names the commands that can never work over MCP — the
full-screen TUIs and the endless streams — through ``exclude``; they are left out of
the typed tools and refused by the run escape hatch.

Two hand-written tools remain as an escape hatch when ``escape_hatches`` is on:

* ``{app}_help`` — walks the command tree via ``--help`` so the model can discover
  details the generated schema does not carry.
* ``{app}_run`` — executes an arbitrary command line, for anything the typed tools
  do not cover.

Commands run as subprocesses, which keeps heavy imports and any crash outside this
process. Nothing here bounds how long a command may run: how long to wait is the MCP
client's decision, and this server never kills a call of its own accord. The one case
it must refuse rather than run is a command that can never finish at all — a
full-screen terminal UI would hang until the client abandoned the call — so
``{app}_run`` rejects an excluded path *before* spawning anything; see
:func:`_excluded_path`.

The full tool list costs a large amount of context, so ``--groups`` narrows the
typed tools to the command subtrees named::

    {app}-mcp --groups browse,search,tools.creator

The two escape hatches are left unfiltered on purpose — they exist to reach what the
typed tools do not cover, and each costs a single tool's worth of schema.
"""

from __future__ import annotations

import argparse
import logging
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import click
from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.mcpserver.tools import Tool

from click2mcp._output import compact_json
from click2mcp.generator import generate_tools, resolve_tool_prefix

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

logger = logging.getLogger(__name__)

# ── instructions ───────────────────────────────────────────────────────────

_INSTRUCTIONS_TEMPLATE = """\
Wraps the `{app}` CLI as one typed tool per leaf command: `{prefix}<path>`, with
`_` for every space and hyphen, so `{app} browse following users` is
`{prefix}browse_following_users`. Each typed tool names its own arguments, types,
defaults, and allowed values, and its description ends with the exact command line
it maps to.

Prefer the typed tools: they are checked against the CLI's own parameters before
anything runs.
"""

_ESCAPE_HATCH_INSTRUCTIONS = """
Two escape hatches remain:
- `{app}_help(path)` prints the CLI's own `--help` for any command path, for
  detail the typed schema does not carry.
- `{app}_run(args)` runs an arbitrary command line, for anything the typed tools do
  not cover.
"""

_NO_TIMEOUT_INSTRUCTIONS = """
How long a command may run is the client's decision, not this server's: a call
returns when the command finishes, and nothing here kills it. Commands that can
never finish over MCP — full-screen terminal UIs and endless streams — are refused
before they run when they are listed in `exclude`; do not retry them.
"""


# ── launcher resolution and invocation helpers ─────────────────────────────


class ClickMCP:
    """Expose a Click CLI as a typed MCP server.

    :param app: the CLI executable name; also the default tool-name prefix
        (``{app}_{path}``), the MCP server name, and the ``--groups`` parser's
        ``prog`` prefix (``{app}-mcp``).
    :param cli: the root :class:`click.Group`, a zero-argument callable returning
        one (or ``None``), or ``None``. A callable is resolved once, and any
        exception while resolving it costs the typed tools but not the server.
    :param module: module name for the ``python -m`` launcher fallback.
    :param instructions: server instructions; a generic template when ``None``.
    :param exclude: command paths to leave out of the typed tools and refuse in
        ``{app}_run``.
    :param include: command path prefixes to expose; ``None`` for all.
    :param escape_hatches: whether to register ``{app}_help`` and ``{app}_run``.
    :param help_description: description for ``{app}_help``; generic when ``None``.
    :param run_description: description for ``{app}_run``; generic when ``None``.
    :param inherit_group_options: whether options declared on ancestor groups are
        exposed on each leaf tool and emitted before the sub-command they lead to.
    :param tool_prefix: prefix for the generated typed tool names. ``None`` keeps
        ``f"{app}_"``; ``""`` drops the prefix entirely; any other value is
        normalized by :func:`resolve_tool_prefix`. The ``{app}_help`` and
        ``{app}_run`` escape hatches always keep the ``{app}_`` prefix: ``help``
        and ``run`` are generic verbs that would collide with real subcommands of
        the same name.
    :param launcher: explicit argv prefix used to run the CLI; when ``None``, the
        sibling console script, then ``$PATH``, then ``python -m module`` is tried.
    :param env: extra environment variables for the child, merged over the
        defaults ``NO_COLOR=1``, ``TERM=dumb``, and ``PYTHONIOENCODING=utf-8``.
    :param invoke: custom subprocess runner ``(tokens) -> CompletedProcess[str]``.
        When given, the escape hatches and generated tools both use it.
    :raises RuntimeError: at construction when no launcher can be resolved and no
        custom ``invoke`` was supplied.
    :raises ValueError: at construction when ``tool_prefix`` does not normalize to
        a valid Python identifier.
    """

    def __init__(
        self,
        *,
        app: str,
        cli: click.Group | Callable[[], click.Group | None] | None,
        module: str | None = None,
        instructions: str | None = None,
        exclude: frozenset[tuple[str, ...]] = frozenset(),
        include: tuple[tuple[str, ...], ...] | None = None,
        escape_hatches: bool = True,
        help_description: str | None = None,
        run_description: str | None = None,
        inherit_group_options: bool = True,
        tool_prefix: str | None = None,
        launcher: Sequence[str] | None = None,
        env: Mapping[str, str] | None = None,
        invoke: Callable[[list[str]], subprocess.CompletedProcess[str]] | None = None,
    ) -> None:
        self.app = app
        # Resolved once here, not per tool: the prefix is fixed for the server's life.
        self.tool_prefix = resolve_tool_prefix(app, tool_prefix)
        self.module = module
        self.exclude = frozenset(exclude)
        self.include = include
        self.escape_hatches = escape_hatches
        self.inherit_group_options = inherit_group_options
        self._launcher = list(launcher) if launcher is not None else None
        self._env = dict(env or {})

        if cli is not None and not isinstance(cli, click.Group) and callable(cli):
            try:
                cli = cli()
            except Exception:
                logger.warning(
                    f"{self.app}-mcp: could not load the CLI; serving without typed tools",
                    exc_info=True,
                )
                cli = None
        self.cli = cli

        self.instructions = instructions if instructions is not None else self._default_instructions()
        self.help_description = help_description if help_description is not None else self._default_help_description()
        self.run_description = run_description if run_description is not None else self._default_run_description()

        # Resolve the launcher once, so a missing executable is a clear failure here
        # rather than at the first tool call. A custom runner needs no launcher.
        self._argv_prefix = [] if invoke is not None else self._resolve_launcher()
        self._runner = invoke if invoke is not None else self._invoke

        # Built eagerly, like the reference module-level ``mcp = _build_server()``.
        self.server = self.build_server(self.include)

    # ── defaults ───────────────────────────────────────────────────────────

    def _default_instructions(self) -> str:
        """Build the generic server instructions.

        :returns: the template for the typed tools, using the resolved prefix so
            the example names are the ones the server actually exposes, plus the
            escape-hatch section only when those tools are enabled.
        """
        text = _INSTRUCTIONS_TEMPLATE.format(app=self.app, prefix=self.tool_prefix)
        if self.escape_hatches:
            text += _ESCAPE_HATCH_INSTRUCTIONS.format(app=self.app)
            text += _NO_TIMEOUT_INSTRUCTIONS.format(app=self.app)
        return text

    def _default_help_description(self) -> str:
        """Build the default description for the help escape hatch.

        :returns: the description text.
        """
        return (
            f"Show the {self.app} CLI command tree or a command's options. "
            "Call with no argument to list the top-level command groups, then drill in."
        )

    def _default_run_description(self) -> str:
        """Build the default description for the run escape hatch.

        :returns: the description text.
        """
        return (
            f"Run a {self.app} CLI command and return its output. `args` is the command line "
            f"without the leading `{self.app}`; quote arguments that contain spaces. "
            "Commands that print styled text are still usable; their output is returned as-is."
        )

    # ── launcher resolution ────────────────────────────────────────────────

    def _resolve_launcher(self) -> list[str]:
        """Return the argv prefix that invokes the CLI.

        Uses an explicit ``launcher`` verbatim, then the console script next to the
        running interpreter (guaranteeing the same environment and a correct
        ``prog_name`` in ``--help`` output), then ``$PATH``, then ``python -m``.

        :returns: argv prefix to prepend to the CLI arguments.
        :raises RuntimeError: if no executable is found and no ``module`` was given.
        """
        if self._launcher is not None:
            return list(self._launcher)
        sibling = Path(sys.executable).parent / self.app
        if sibling.is_file() and os.access(sibling, os.X_OK):
            return [str(sibling)]
        found = shutil.which(self.app)
        if found:
            return [found]
        if self.module:
            return [sys.executable, "-m", self.module]
        raise RuntimeError(
            f"{self.app}-mcp: cannot find the `{self.app}` executable and no module was given; "
            f"pass launcher=[...] or module='...'."
        )

    def _invoke(self, tokens: list[str]) -> subprocess.CompletedProcess[str]:
        """Run the CLI with ``tokens`` and capture its output.

        Nothing here bounds how long the command may run: it returns when it exits,
        and deciding when to give up on it is the client's call, not this server's.

        :param tokens: CLI arguments, already split.
        :returns: the completed process.
        """
        env = {
            **os.environ,
            "NO_COLOR": "1",
            "TERM": "dumb",
            "PYTHONIOENCODING": "utf-8",
            **self._env,
        }
        return subprocess.run(
            [*self._argv_prefix, *tokens],
            capture_output=True,
            text=True,
            env=env,
            # This process's own stdin is the JSON-RPC pipe when serving over stdio,
            # so the child is given /dev/null rather than a handle onto the protocol
            # stream.
            stdin=subprocess.DEVNULL,
        )

    # ── hand-written escape hatches ────────────────────────────────────────

    def _help_tool(self) -> Tool:
        """Build the ``{app}_help`` tool.

        Only ``ToolError`` carries its message to the client: the SDK turns any
        other exception into a generic ``UnexpectedToolError`` and discards the text.

        :returns: the built tool.
        """
        app = self.app

        def help_fn(path: str = "") -> str:
            """Show the CLI command tree or a command's options.

            Call with no argument to list the top-level command groups, then drill in.

            :param path: space-separated command path, e.g. ``"search"`` or
                ``"search question"``. Empty for the top level.
            :returns: the CLI's help text.
            :raises ToolError: if the command exits non-zero.
            """
            tokens = shlex.split(path)
            command = f"{app} {' '.join(tokens)} --help".strip()
            proc = self._runner([*tokens, "--help"])
            if proc.returncode != 0:
                raise ToolError(
                    f"`{command}` exited with code {proc.returncode}.\n{proc.stderr.strip() or '(no stderr)'}"
                )
            return proc.stdout

        help_fn.__name__ = f"{app}_help"
        return Tool.from_function(help_fn, name=f"{app}_help", description=self.help_description)

    def _run_tool(self) -> Tool:
        """Build the ``{app}_run`` tool.

        Only ``ToolError`` carries its message to the client: the SDK turns any
        other exception into a generic ``UnexpectedToolError`` and discards the text.

        :returns: the built tool.
        """
        app = self.app

        def run_fn(args: str) -> str:
            """Run a CLI command and return its output.

            :param args: the command line to run, without the leading app name.
                Quote arguments that contain spaces.
            :returns: the command's stdout.
            :raises ToolError: if no command is given, if the command is one that can
                never finish over MCP (a full-screen terminal UI or an endless
                stream), or if it exits non-zero.
            """
            tokens = shlex.split(args)
            if not tokens:
                raise ToolError(f"No command given. Call {app}_help() to discover commands.")

            blocked = _excluded_path(tokens, self.exclude)
            if blocked is not None:
                raise ToolError(
                    f"`{app} {' '.join(blocked)}` opens a full-screen terminal UI (or streams until "
                    f"interrupted) and can never run over MCP: nothing here can display it or stop it, "
                    f"so the call would hang until it was abandoned. Nothing was run — ask the user to "
                    f"run it in their own terminal instead."
                )

            proc = self._runner(tokens)
            if proc.returncode != 0:
                raise ToolError(
                    f"`{app} {' '.join(tokens)}` exited with code {proc.returncode}.\n"
                    f"{proc.stderr.strip() or '(no stderr)'}"
                )

            # Successful commands often warn on stderr; only surface it when stdout is
            # empty, so that JSON output stays clean and parseable.
            if not proc.stdout.strip() and proc.stderr.strip():
                return compact_json(proc.stderr.strip())

            return compact_json(proc.stdout)

        run_fn.__name__ = f"{app}_run"
        return Tool.from_function(run_fn, name=f"{app}_run", description=self.run_description)

    # ── server construction and entry point ────────────────────────────────

    def build_server(self, include: tuple[tuple[str, ...], ...] | None = None) -> MCPServer:
        """Build an MCP server exposing the selected command subtrees.

        One typed tool per leaf command is generated from the Click tree here rather
        than registered afterwards: ``MCPServer.add_tool`` derives the JSON Schema
        from a function signature, so the tools have to be built before the
        constructor runs. Generation is best-effort — a CLI that could not be loaded
        costs the typed tools, not the server.

        :param include: command path prefixes to expose, or ``None`` to fall back to
            the constructor's ``include``. The escape hatches are registered either
            way when enabled.
        :returns: the configured server.
        """
        effective = self.include if include is None else include
        tools = generate_tools(
            cli=self.cli,
            invoke=self._runner,
            app=self.app,
            exclude=self.exclude,
            include=effective,
            inherit_group_options=self.inherit_group_options,
            tool_prefix=self.tool_prefix,
        )
        if self.escape_hatches:
            tools.append(self._help_tool())
            tools.append(self._run_tool())
        return MCPServer(name=self.app, instructions=self.instructions, tools=tools)

    def run(self, argv: Sequence[str] | None = None) -> None:
        """Serve MCP over stdio, optionally restricted to selected command groups.

        :param argv: command-line arguments; ``sys.argv[1:]`` when ``None``.
        """
        unfiltered = " The escape hatches stay unfiltered." if self.escape_hatches else ""
        parser = argparse.ArgumentParser(
            prog=f"{self.app}-mcp",
            description=f"Expose the {self.app} CLI to AI agents over MCP (stdio).",
        )
        parser.add_argument(
            "--groups",
            metavar="PATHS",
            type=_group_prefixes,
            help=(
                "comma-separated command paths to expose as typed tools, e.g. "
                "'browse,search,tools.creator'; each path selects its whole subtree. "
                f"Omit to expose every command.{unfiltered}"
            ),
        )
        args = parser.parse_args(argv)
        server = self.server if args.groups is None else self.build_server(args.groups)
        server.run(transport="stdio")


def _group_prefixes(value: str) -> tuple[tuple[str, ...], ...]:
    """Parse the ``--groups`` argument into command path prefixes.

    Each comma-separated entry names a command path, written with dots
    (``tools.creator``) or spaces (``tools creator``). The path is a prefix, so
    ``tools`` selects the whole ``tools`` subtree; an empty entry is refused
    instead of silently selecting everything.

    :param value: the raw argument value.
    :returns: the path prefixes, in the order given.
    :raises argparse.ArgumentTypeError: if an entry is empty.
    """
    prefixes: list[tuple[str, ...]] = []
    for entry in value.split(","):
        tokens = tuple(entry.replace(".", " ").split())
        if not tokens:
            raise argparse.ArgumentTypeError(f"empty command path in {value!r}")
        prefixes.append(tokens)
    return tuple(prefixes)


def _excluded_path(tokens: list[str], exclude: frozenset[tuple[str, ...]]) -> tuple[str, ...] | None:
    """Find the excluded command path ``tokens`` starts with, if any.

    Nothing kills a subprocess, so a command that opens a full-screen terminal UI —
    or streams until interrupted — would hang until the client abandoned the call.
    Refusing it up front is the only cheap answer, and it costs no subprocess.

    The match is a plain comparison of the leading tokens against each excluded
    path, so an option that trails the path (``chat interactive --foo``) still
    matches, while one that precedes it (``--profile x chat interactive``) does not.
    That is the intended trade: these commands are reached by their bare path, and
    the generated tools cannot hit this at all, since those paths are never
    generated.

    :param tokens: CLI arguments, already split.
    :param exclude: the excluded command paths.
    :returns: the excluded path that was asked for, or ``None``.
    """
    for path in exclude:
        if tuple(tokens[: len(path)]) == path:
            return path
    return None
