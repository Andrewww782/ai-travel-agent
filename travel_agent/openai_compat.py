"""Adapter for any provider with an OpenAI-style chat-completions API (Gemini, Groq, OpenRouter,
Ollama and others), so the agent can run on a model other than Claude.

The agent loop speaks one shape: Anthropic-style messages in, a response with `content` blocks,
`stop_reason` and `usage` out. This module translates that shape to chat-completions and back.
Nothing else in the project knows which provider is in use.

Uses only the standard library, so it adds no dependency.
"""
import asyncio
import json
import time
import urllib.error
import urllib.request
from types import SimpleNamespace

STOP_REASONS = {"tool_calls": "tool_use", "function_call": "tool_use", "length": "max_tokens", "content_filter": "refusal"}
RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}


class ProviderError(RuntimeError):
    def __init__(self, status: int, body: str):
        super().__init__(f"provider returned {status}: {body[:300]}")
        self.status = status


def to_chat_messages(system: str, messages: list[dict]) -> list[dict]:
    out = [{"role": "system", "content": system}]
    for message in messages:
        content = message["content"]
        if isinstance(content, str):
            out.append({"role": message["role"], "content": content})
        elif message["role"] == "assistant":
            # Replay the provider's own message untouched: some models attach fields to their tool
            # calls that must come back exactly as sent.
            raw = next((b.message for b in content if b.type == "provider_raw"), None)
            if raw is None:
                raise ValueError("assistant turn was not produced by this adapter; switch provider only between turns")
            out.append(raw)
        else:  # a user turn holding tool results: chat-completions wants one "tool" message per result
            for block in content:
                out.append({"role": "tool", "tool_call_id": block["tool_use_id"], "content": block["content"]})
    return out


def to_chat_tools(tool_specs: list[dict]) -> list[dict]:
    return [
        {"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]}}
        for t in tool_specs
    ]


def from_chat_response(data: dict):
    choice = data["choices"][0]
    message = choice["message"]
    blocks = []
    if message.get("content"):
        blocks.append(SimpleNamespace(type="text", text=message["content"]))
    for call in message.get("tool_calls") or []:
        try:
            args = json.loads(call["function"].get("arguments") or "{}")
        except json.JSONDecodeError:
            args = None
        # Arguments that are not a JSON object become {}: the tool then rejects the call with a
        # message naming the missing fields, which the model can read and correct.
        blocks.append(
            SimpleNamespace(type="tool_use", id=call["id"], name=call["function"]["name"], input=args if isinstance(args, dict) else {})
        )
    blocks.append(SimpleNamespace(type="provider_raw", message={"role": "assistant", **{k: v for k, v in message.items() if k != "role"}}))
    has_calls = any(b.type == "tool_use" for b in blocks)
    usage = data.get("usage") or {}
    return SimpleNamespace(
        content=blocks,
        # Some providers report "stop" even when they return tool calls, so the calls decide.
        stop_reason="tool_use" if has_calls else STOP_REASONS.get(choice.get("finish_reason"), "end_turn"),
        usage=SimpleNamespace(input_tokens=usage.get("prompt_tokens", 0), output_tokens=usage.get("completion_tokens", 0)),
    )


def _post(url: str, api_key: str, payload: dict, timeout: float, attempts: int) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as err:
            body = err.read().decode("utf-8", "replace")
            if err.code not in RETRY_STATUS or attempt == attempts:
                raise ProviderError(err.code, body) from None
            # Free tiers rate-limit hard: wait as long as the provider asks, else back off 2, 4, 8... seconds.
            retry_after = err.headers.get("Retry-After", "")
            wait = float(retry_after) if retry_after.replace(".", "", 1).isdigit() else 2.0**attempt
        except (urllib.error.URLError, TimeoutError) as err:
            if attempt == attempts:
                raise ProviderError(0, f"could not reach the provider: {err}") from None
            wait = 2.0**attempt
        time.sleep(min(wait, 65))
    raise AssertionError("unreachable")


def make_llm(base_url: str, api_key: str, model: str, min_interval: float = 0.0, attempts: int = 2, timeout: float = 60.0):
    """Returns `async llm(system, tools, messages)` with the same contract as agent.make_llm.

    min_interval spaces requests at least that many seconds apart, for free tiers that allow only a
    few requests per minute. attempts is how many times one request is tried: keep it low for a
    person waiting in a chat, raise it for a batch job like the evals that can afford to wait.
    """
    url = base_url.rstrip("/") + "/chat/completions"
    gate, last = asyncio.Lock(), [0.0]

    async def llm(system: str, tool_specs: list[dict], messages: list[dict]):
        payload = {"model": model, "messages": to_chat_messages(system, messages)}
        if tool_specs:
            payload["tools"] = to_chat_tools(tool_specs)
        async with gate:  # one request at a time, so the spacing holds across concurrent eval cases
            pause = last[0] + min_interval - time.monotonic()
            if pause > 0:
                await asyncio.sleep(pause)
            try:
                return from_chat_response(await asyncio.to_thread(_post, url, api_key, payload, timeout, attempts))
            finally:
                last[0] = time.monotonic()

    return llm
