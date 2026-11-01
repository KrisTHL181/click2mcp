"""Shape command output for an MCP client.

Pretty-printed JSON is a courtesy to a human reading a terminal. A model pays for
every indentation byte and learns nothing from it, yet some CLIs pretty-print their
``--json`` output by default. :func:`compact_json` re-serializes such output onto a
single line with minimal separators, and leaves anything that is not a single JSON
document exactly as the CLI wrote it.
"""

from __future__ import annotations

import json


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
