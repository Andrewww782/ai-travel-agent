"""AI Travel Agent: a tool-using LLM agent with confirm-gated writes, MCP tools and evals."""
import os
from pathlib import Path

# Read KEY=VALUE lines from a local .env file, without overriding real environment variables.
# shortcut: no quoting or multi-line values; use python-dotenv if the file ever needs them.
if Path(".env").is_file():
    for _line in Path(".env").read_text(encoding="utf-8").splitlines():
        _key, _sep, _value = _line.partition("=")
        if _sep and _key.strip() and not _key.lstrip().startswith("#"):
            os.environ.setdefault(_key.strip(), _value.strip())
