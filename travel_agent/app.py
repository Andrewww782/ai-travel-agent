"""The web API and chat page. Run with:  uvicorn travel_agent.app:create_app --factory

Each visitor gets an anonymous user tied to a signed cookie. The user id is always read from that
cookie on the server; nothing the browser or the model sends can choose a different user.
"""
import asyncio
import hashlib
import hmac
import logging
import os
import secrets
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path

import anthropic
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse
from mcp import Client
from pydantic import BaseModel, Field

from . import agent, db, tools
from .openai_compat import ProviderError
from .mcp_server import build_server

log = logging.getLogger("travel_agent")
COOKIE = "session"


class SessionIn(BaseModel):
    code: str = Field(default="", max_length=200)


class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=1000)


class DecisionIn(BaseModel):
    confirm: bool


def create_app(db_path: str | None = None, llm=None) -> FastAPI:
    db_path = db_path or os.environ.get("DB_PATH", "travel.db")
    # shortcut: without SECRET_KEY a new key is made at start-up, so sessions end on restart; set it in production.
    secret = (os.environ.get("SECRET_KEY") or secrets.token_hex(32)).encode()
    demo_code = os.environ.get("DEMO_CODE", "")  # when set, visitors must enter it before they can chat
    user_daily = int(os.environ.get("USER_DAILY_MESSAGES", "30"))
    global_daily = int(os.environ.get("GLOBAL_DAILY_MESSAGES", "300"))
    secure_cookie = os.environ.get("COOKIE_SECURE", "") == "1"  # set to 1 when served over https
    state: dict = {}
    locks: defaultdict[int, asyncio.Lock] = defaultdict(asyncio.Lock)  # one turn at a time per user

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        state["conn"] = db.init(db_path)
        async with Client(build_server(db_path)) as client:
            state["toolbox"] = await agent.Toolbox(client).load()
            state["llm"] = llm or agent.make_llm()
            yield
        state["conn"].close()

    app = FastAPI(title="AI Travel Agent", lifespan=lifespan, docs_url=None, redoc_url=None)

    def sign(user_id: int) -> str:
        return f"{user_id}.{hmac.new(secret, str(user_id).encode(), hashlib.sha256).hexdigest()}"

    def current_user(request: Request) -> int | None:
        user_id, _, mac = request.cookies.get(COOKIE, "").partition(".")
        if not user_id.isdigit() or not hmac.compare_digest(sign(int(user_id)), f"{user_id}.{mac}"):
            return None
        return int(user_id)

    def require_user(request: Request) -> int:
        user_id = current_user(request)
        if user_id is None:
            raise HTTPException(401, "Start a session first.")
        return user_id

    def view(user_id: int) -> dict:
        conn = state["conn"]
        q = lambda sql: conn.execute(sql, (user_id,)).fetchall()  # noqa: E731
        return {
            "signed_in": True,
            "messages": [dict(r) for r in q("SELECT role, text FROM messages WHERE user_id = ? ORDER BY id")],
            "pending": [{"id": p["id"], "summary": p["summary"]} for p in agent.pending_for(conn, user_id)],
            "preferences": [r["text"] for r in q("SELECT text FROM preferences WHERE user_id = ? ORDER BY id")],
            "trips": tools.list_trips(conn, user_id),
        }

    @app.get("/")
    def page():
        return FileResponse(Path(__file__).parent / "static" / "index.html")

    @app.get("/healthz")
    def health():
        return {"ok": True}

    @app.get("/api/state")
    def get_state(request: Request):
        user_id = current_user(request)
        return view(user_id) if user_id else {"signed_in": False, "needs_code": bool(demo_code)}

    @app.post("/api/session")
    def start_session(body: SessionIn, response: Response):
        """Start as a new anonymous visitor. Also used by the page's "new visitor" button."""
        if demo_code and not hmac.compare_digest(body.code.encode(), demo_code.encode()):
            raise HTTPException(403, "That access code is not right.")
        user_id = db.new_user(state["conn"])
        response.set_cookie(
            COOKIE, sign(user_id), httponly=True, samesite="lax", secure=secure_cookie, max_age=30 * 86400
        )
        return view(user_id)

    @app.post("/api/chat")
    async def chat(body: ChatIn, request: Request):
        user_id, conn = require_user(request), state["conn"]
        # Cost guard for a public demo: the API key behind this endpoint is paid for by the owner.
        today = "SELECT COUNT(*) FROM messages WHERE role = 'user' AND created_at >= date('now')"
        if conn.execute(today + " AND user_id = ?", (user_id,)).fetchone()[0] >= user_daily:
            raise HTTPException(429, "You have reached today's message limit for this demo.")
        if conn.execute(today).fetchone()[0] >= global_daily:
            raise HTTPException(429, "This demo has reached its message limit for today.")
        async with locks[user_id]:
            try:
                turn = await agent.run_turn(state["llm"], state["toolbox"], conn, user_id, body.message.strip())
            except TypeError as err:
                # With no credentials at all, the SDK raises a TypeError before sending anything.
                if "authentication" not in str(err):
                    raise
                raise HTTPException(503, "No model API key is configured on this server.") from None
            except anthropic.AuthenticationError:
                raise HTTPException(503, "The server's model API key is invalid.") from None
            except anthropic.RateLimitError:
                raise HTTPException(503, "The model is busy. Try again in a moment.") from None
            except anthropic.APIStatusError as err:
                log.error("model API error %s: %s", err.status_code, err.message)
                raise HTTPException(502, "The model returned an error. Try again.") from None
            except anthropic.APIConnectionError:
                raise HTTPException(502, "Could not reach the model. Try again.") from None
            except ProviderError as err:  # the OpenAI-compatible path, after its own retries
                log.error("%s", err)
                if err.status in (429, 503):  # rate-limited, or the provider says the model is overloaded
                    raise HTTPException(503, "The model is busy right now. Try again in a moment.") from None
                raise HTTPException(502, "The model returned an error. Try again.") from None
        log.info(
            "turn user=%s steps=%s tools=%s tokens=%s/%s",
            user_id, turn.steps, [t["tool"] for t in turn.trace], turn.input_tokens, turn.output_tokens,
        )  # fmt: skip
        return view(user_id)

    @app.post("/api/pending/{pending_id}")
    async def decide(pending_id: int, body: DecisionIn, request: Request):
        user_id = require_user(request)
        async with locks[user_id]:
            message = await agent.resolve_pending(state["toolbox"], state["conn"], user_id, pending_id, body.confirm)
        if message is None:  # same answer for "not yours" and "does not exist"
            raise HTTPException(404, "That request is no longer waiting for confirmation.")
        return view(user_id)

    @app.post("/api/reset")
    def reset(request: Request):
        """Clear the conversation. Saved trips and preferences stay."""
        user_id, conn = require_user(request), state["conn"]
        conn.execute("UPDATE pending SET status = 'cancelled' WHERE user_id = ? AND status = 'pending'", (user_id,))
        conn.execute("DELETE FROM messages WHERE user_id = ?", (user_id,))
        conn.commit()
        return view(user_id)

    return app
