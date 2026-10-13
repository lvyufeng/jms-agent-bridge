"""``python -m jms_bridge`` — the CLI, for a checkout that was never ``pip install``-ed.

``[project.scripts]`` puts ``jms-bridge`` on ``PATH`` only after an install, so a freshly
cloned repo had no way to reach ``init`` / ``login`` / ``doctor`` short of
``python -c 'from jms_bridge.cli import main; ...'``. This is the same ``main()``, reached
the ordinary way.

The command is ``serve`` for the MCP server; ``jms-bridge`` with no arguments prints usage.
"""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
