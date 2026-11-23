"""The OpenAI-compatible adapter, against a fake provider running on localhost."""
import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from travel_agent import agent, openai_compat
from travel_agent.agent import run_turn

from .helpers import run


def answer(content=None, tool_calls=None, finish="stop"):
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {"choices": [{"message": message, "finish_reason": finish}], "usage": {"prompt_tokens": 100, "completion_tokens": 20}}


def call(name, args, call_id="call_1", **extra):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": args}, **extra}


@pytest.fixture
def provider():
    """Returns (base_url, queue, seen). Queue items are (status, body, headers); seen holds the requests."""
    queue, seen = [], []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append({"path": self.path, "auth": self.headers["Authorization"], "body": body})
            status, reply, headers = queue.pop(0)
            payload = json.dumps(reply).encode()
            self.send_response(status)
            for key, value in {"Content-Type": "application/json", "Content-Length": str(len(payload)), **headers}.items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}/v1/", queue, seen
    server.shutdown()
    server.server_close()


def test_a_whole_turn_through_the_adapter(provider):
    base_url, queue, seen = provider
    # The provider attaches an extra field to its tool call that must come back unchanged.
    first = call("search_flights", '{"origin": "DEL", "destination": "DXB", "date": "2027-03-10"}', extra_content={"signature": "abc"})
    queue += [(200, answer(tool_calls=[first], finish="stop"), {}), (200, answer("Three flights: AI915, EK511, 6E1461."), {})]

    async def body(env):
        llm = openai_compat.make_llm(base_url, "secret-key", "some-model")
        turn = await run_turn(llm, env.toolbox, env.conn, env.alice, "flights to dubai")
        assert turn.reply == "Three flights: AI915, EK511, 6E1461." and turn.steps == 2
        assert turn.trace[0]["tool"] == "search_flights" and not turn.trace[0]["is_error"]
        assert (turn.input_tokens, turn.output_tokens) == (200, 40)

    run(body)
    one, two = seen[0]["body"], seen[1]["body"]
    assert seen[0]["path"] == "/v1/chat/completions" and seen[0]["auth"] == "Bearer secret-key"
    assert one["model"] == "some-model" and one["messages"][0]["role"] == "system"
    assert one["messages"][-1] == {"role": "user", "content": "flights to dubai"}
    tool = next(t["function"] for t in one["tools"] if t["function"]["name"] == "save_trip")
    assert "user_id" not in tool["parameters"]["properties"]
    assistant, result = two["messages"][-2], two["messages"][-1]
    assert assistant["tool_calls"] == [first] and assistant["role"] == "assistant"
    assert result["role"] == "tool" and result["tool_call_id"] == "call_1" and "AI915" in result["content"]


def test_rate_limit_is_retried_and_a_hard_error_is_raised(provider, monkeypatch):
    base_url, queue, seen = provider
    monkeypatch.setattr(openai_compat.time, "sleep", lambda seconds: None)
    llm = openai_compat.make_llm(base_url, "k", "m", attempts=3)
    queue += [(429, {"error": "slow down"}, {"Retry-After": "0"}), (503, {}, {}), (200, answer("ok"), {})]
    response = asyncio.run(llm("sys", [], [{"role": "user", "content": "hi"}]))
    assert response.content[0].text == "ok" and response.stop_reason == "end_turn" and len(seen) == 3
    assert "tools" not in seen[0]["body"]
    queue.append((401, {"error": "bad key"}, {}))
    with pytest.raises(openai_compat.ProviderError, match="401") as err:
        asyncio.run(llm("sys", [], [{"role": "user", "content": "hi"}]))
    assert err.value.status == 401 and len(seen) == 4  # not retried
    # the default gives up after two tries, so a person in the chat is not left waiting for minutes
    quick = openai_compat.make_llm(base_url, "k", "m")
    queue += [(503, {}, {}), (503, {}, {}), (200, answer("never reached"), {})]
    with pytest.raises(openai_compat.ProviderError, match="503"):
        asyncio.run(quick("sys", [], [{"role": "user", "content": "hi"}]))
    assert len(seen) == 6


@pytest.mark.parametrize("finish, expected", [("length", "max_tokens"), ("content_filter", "refusal"), ("stop", "end_turn"), (None, "end_turn")])
def test_finish_reasons_map_to_the_stop_reasons_the_agent_knows(finish, expected):
    assert openai_compat.from_chat_response(answer("text", finish=finish)).stop_reason == expected


def test_unreadable_tool_arguments_become_an_empty_call_the_tool_will_reject():
    calls = [call("save_trip", "{not json"), call("save_trip", "[1]", "call_2")]
    response = openai_compat.from_chat_response(answer(tool_calls=calls, finish="tool_calls"))
    assert [b.input for b in response.content if b.type == "tool_use"] == [{}, {}]
    assert response.stop_reason == "tool_use"


def test_provider_is_chosen_from_the_environment(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:1/v1")
    monkeypatch.delenv("AGENT_MODEL", raising=False)
    with pytest.raises(RuntimeError, match="AGENT_MODEL"):
        agent.make_llm()
    monkeypatch.setenv("AGENT_MODEL", "x")
    assert agent.make_llm("x").__module__ == "travel_agent.openai_compat"
