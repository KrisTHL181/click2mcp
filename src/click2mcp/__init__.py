"""Turn any Click CLI into a typed MCP server.

The server exposes one tool per leaf command, generated from the Click tree at
startup. The schema has to come from a *function signature*: ``MCPServer.add_tool``
and :meth:`Tool.from_function` derive the JSON Schema from the signature through
pydantic and offer no seam for a hand-written ``inputSchema``. Each generated tool
therefore carries a synthesized signature built from its command's own parameters,
with the CLI's real names, types, defaults, and enums.
"""

from __future__ import annotations

from click2mcp.generator import ParamSpec, ToolSpec, generate_tools
from click2mcp.server import ClickMCP

__all__ = ["ClickMCP", "ParamSpec", "ToolSpec", "generate_tools"]
