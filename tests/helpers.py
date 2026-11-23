"""Test helpers: a real database and a real in-process MCP server, with a scripted fake model."""
import asyncio
import tempfile
from pathlib import Path
from types import SimpleNamespace

from mcp import Client

from travel_agent import db
from travel_agent.agent import Toolbox
from travel_agent.mcp_server import build_server

USAGE = SimpleNamespace(input_tokens=10, output_tokens=5)


def say(text):
    return SimpleNamespace(type="text", text=text)


def use(name, _id="t1", **args):
    return SimpleNamespace(type="tool_use", id=_id, name=name, input=args)


def reply(*blocks, stop=None):
    stop = stop or ("tool_use" if any(b.type == "tool_use" for b in blocks) else "end_turn")
    return SimpleNamespace(content=list(blocks), stop_reason=stop, usage=USAGE)


def script(*responses):
    """A fake model that returns the given responses in order and records what it was sent."""
    queue, calls = list(responses), []

    async def llm(system, tools, messages):
        calls.append({"system": system, "tools": tools, "messages": list(messages)})
        return queue.pop(0) if len(queue) > 1 else queue[0]  # the last response repeats

    llm.calls = calls
    return llm


def run(body):
    """Run `async body(env)` against a fresh database. env has conn, toolbox, path and two users."""

    async def main():
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
            path = str(Path(folder) / "test.db")
            conn = db.init(path)
            try:
                async with Client(build_server(path)) as client:
                    env = SimpleNamespace(conn=conn, path=path, toolbox=await Toolbox(client).load())
                    env.alice, env.bob = db.new_user(conn), db.new_user(conn)
                    return await body(env)
            finally:
                conn.close()

    return asyncio.run(main())


def count(conn, table, user_id):
    return conn.execute(f"SELECT COUNT(*) FROM {table} WHERE user_id = ?", (user_id,)).fetchone()[0]
