# click2mcp

`click2mcp` turns a [Click](https://click.palletsprojects.com/) CLI into a typed
MCP server. At startup it walks the command tree and creates one MCP tool per leaf
command, using the command's parameter names, types, defaults, and allowed values.

The JSON Schema comes from a function signature. `MCPServer.add_tool` and
`Tool.from_function` build it with pydantic and do not accept a hand-written
`inputSchema`, so `click2mcp` gives each command a function whose `__signature__`
and `__annotations__` are the Click parameters themselves.

## Install

```bash
pip install click2mcp
```

## Quickstart

Add an adapter module to your CLI package:

```python
# src/mycli/mcp/server.py
from click2mcp import ClickMCP
from mycli.main import main as cli

bridge = ClickMCP(app="mycli", cli=cli, module="mycli.main")
mcp = bridge.server


def main() -> None:
    bridge.run()
```

Expose it as a console script and an optional dependency:

```toml
[project.scripts]
mycli-mcp = "mycli.mcp.server:main"

[project.optional-dependencies]
mcp = ["click2mcp>=0.1.0"]
```

Install the extra, register the server with an MCP client, and run it over stdio.
For Claude Code:

```bash
claude mcp add mycli -- mycli-mcp
```

`cli` can also be a zero-argument callable that returns the root group or `None`.
It is resolved once. If it raises, the server still starts with only the escape
hatches. `module="mycli.main"` is the fallback launcher used when no installed
`mycli` executable can be found.

## Configuration

| Parameter | Default | Meaning |
| --- | --- | --- |
| `app` | required | CLI executable name, tool-name prefix (`{app}_{path}`), MCP server name, and `{app}-mcp` program name. |
| `cli` | required | Root `click.Group`, zero-arg callable returning one, or `None`. |
| `module` | `None` | Module passed to `python -m` when no executable is found. Required if `app` is not installed and no `launcher` is given. |
| `instructions` | generic template | MCP server instructions; used verbatim when supplied. |
| `exclude` | `frozenset()` | Command paths omitted from the typed tools and refused by `{app}_run`. |
| `include` | `None` | Command path prefixes to expose; `None` exposes every leaf outside `exclude`. |
| `escape_hatches` | `True` | Register `{app}_help` and `{app}_run`. |
| `help_description` | generic | Description override for `{app}_help`. |
| `run_description` | generic | Description override for `{app}_run`. |
| `inherit_group_options` | `True` | Expose options declared on ancestor groups on every leaf tool below them. |
| `launcher` | `None` | Explicit argv prefix used to run the CLI. |
| `env` | `None` | Extra child environment variables, merged over the defaults. |
| `invoke` | `None` | Custom `(tokens) -> CompletedProcess[str]` runner; shared by typed tools and escape hatches. |

The launcher is resolved once, in this order: `launcher` verbatim, the console
script next to the running interpreter, `shutil.which(app)`, then
`[sys.executable, "-m", module]`. If none of them exist, construction raises
`RuntimeError`. The child process gets `NO_COLOR=1`, `TERM=dumb`, and
`PYTHONIOENCODING=utf-8`, merged over `os.environ` and then over `env`.

## Behavior notes

**Tool generation.** Every leaf command becomes one tool. Spaces and hyphens in the
command path turn into `_`, so `mycli browse following users` is
`mycli_browse_following_users`, and the description ends with the exact CLI
spelling. `--groups` restricts the typed tools to the named subtrees, written with
dots or spaces. An empty entry is rejected.

**Escape hatches and inherited options.** `{app}_help(path)` runs `--help`, and
`{app}_run(args)` runs an arbitrary command line. Set `escape_hatches=False` to omit
both. Only `ToolError` text reaches the client. Typed tools also expose the options
of their ancestor groups, and the generated argv puts each option before the
subcommand it belongs to: `{app} --verbose sub --profile=x run -- n`. A leaf
parameter shadows a group option with the same name, the innermost group wins over
outer ones, and eager options such as `--help` and `--version` are skipped. Set
`inherit_group_options=False` to disable the inheritance.

**Prompts and failures.** An option declared with `prompt=` is exposed as required,
because the child's stdin is `/dev/null` and a prompt can never be answered over
MCP, even when Click does not mark the option required. A leaf whose parameters
cannot be turned into a signature, or whose tool name collides with another
command's, is logged and skipped. The rest of the server still starts.

**JSON output.** Output that is a single JSON document is returned compact — one
line, no indentation or separator padding — because an agent pays for every
indentation token. This is best effort and applies to both the typed tools and
`{app}_run`: help text, logs, JSON Lines, and anything else that is not one complete
JSON document passes through exactly as the CLI wrote it, and key order is preserved.

**Runtime and exclusions.** A call returns when the child exits. There is no
server-side timeout, and the child's stdin is `/dev/null`, so it cannot read the
JSON-RPC stream; deciding when to give up is the MCP client's job. Options are
passed as one token, `--long=value`, never `--long value`, and repeated options are
repeated. Positionals follow an end-of-options `--` in declaration order, so values
that begin with `-` survive. Full-screen TUIs and endless streams cannot work over
MCP. List them in `exclude`, because nothing here kills a subprocess and
`{app}_run` refuses excluded paths before spawning.

## License

MIT
