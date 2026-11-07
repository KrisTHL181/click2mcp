"""Shape command output for an MCP client.

Pretty-printed JSON is a courtesy to a human reading a terminal. A model pays for
every indentation byte and learns nothing from it, yet some CLIs pretty-print their
``--json`` output by default. :func:`compact_json` re-serializes such output onto a
single line with minimal separators, and leaves anything that is not a single JSON
document exactly as the CLI wrote it.

An MCP tool that returns a string gets that string JSON-encoded *inside* the
result, so a JSON document would reach the model with every quote escaped. The
generated tools therefore return :func:`shape_output`, which hands the parsed
object back for a JSON document: the client's structured content then carries the
real object, and only text that is not JSON stays a string.
"""

from __future__ import annotations

import json
from typing import Any

#: What a command's output becomes: the parsed document for a JSON object or
#: array, and the CLI's text otherwise. This is also the return annotation of
#: every generated tool, since MCP derives the output schema from it.
Output = str | dict[str, Any] | list[Any]


def shape_output(text: str) -> Output:
    """Return a command's output as data when it is a JSON document.

    A JSON object or array comes back as the parsed value, so the MCP client
    receives a real object rather than a JSON string needing to be unescaped.
    Everything else — help text, logs, JSON Lines, a banner around JSON, a bare
    scalar — is returned as a string, compacted by :func:`compact_json` where an
    object or array is not involved.

    Only a complete JSON document is treated as data: parsing the whole text is
    what keeps help text and JSON Lines from being rewritten.

    :param text: a command's output.
    :returns: the parsed object or array, otherwise the shaped text.
    """
    stripped = text.strip()
    if not stripped:
        return text
    try:
        value = json.loads(stripped)
    except ValueError:
        return text
    if isinstance(value, dict | list):
        return value
    return compact_json(text)


def compact_json(text: str) -> str:
    """Put a JSON document on one line, when ``text`` is one.

    The text is parsed first, so only complete JSON documents are rewritten: help
    text, logs, JSON Lines, and output with a banner around it pass through
    unchanged, as does empty output. Key order is preserved.

    Re-serialization can normalize a value rather than shorten it (``1e5`` becomes
    ``100000.0``), so the result is only used when it is actually shorter than the
    input. Already-compact JSON therefore comes back untouched.

    :param text: a command's output.
    :returns: the compact JSON, or ``text`` unchanged.
    """
    stripped = text.strip()
    if not stripped:
        return text
    try:
        value = json.loads(stripped)
    except ValueError:
        return text
    compact = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return compact if len(compact) < len(stripped) else text
