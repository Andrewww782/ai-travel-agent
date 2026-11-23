import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from travel_agent.app import create_app

from .helpers import reply, say, script, use

SAVE = (reply(use("save_trip", flight_id="AI915", date="2027-03-10")), reply(say("Waiting for you.")))


@pytest.fixture
def site(monkeypatch):
    """Returns make(llm) -> a fresh browser session against one shared server and database."""
    monkeypatch.setenv("SECRET_KEY", "test-secret")
    for name in ("DEMO_CODE", "USER_DAILY_MESSAGES", "GLOBAL_DAILY_MESSAGES"):
        monkeypatch.delenv(name, raising=False)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
        opened = []

        def make(*responses):
            client = TestClient(create_app(str(Path(folder) / "app.db"), llm=script(*responses or [reply(say("Hi."))])))
            client.__enter__()
            opened.append(client)
            return client

        yield make
        for client in opened:
            client.__exit__(None, None, None)


def test_chat_needs_a_session_and_a_forged_cookie_is_rejected(site):
    browser = site()
    assert browser.post("/api/chat", json={"message": "hi"}).status_code == 401
    browser.post("/api/session", json={})
    assert browser.post("/api/chat", json={"message": "hi"}).json()["messages"][-1] == {"role": "assistant", "text": "Hi."}
    user_id, _, mac = browser.cookies["session"].partition(".")
    browser.cookies.clear()
    forged = {"Cookie": f"session={int(user_id) + 1}.{mac}"}  # try to become the next user
    assert browser.post("/api/chat", json={"message": "hi"}, headers=forged).status_code == 401
    assert browser.get("/api/state", headers=forged).json()["signed_in"] is False
    genuine = {"Cookie": f"session={user_id}.{mac}"}
    assert browser.get("/api/state", headers=genuine).json()["signed_in"] is True


def test_confirm_button_saves_the_trip_and_only_for_its_owner(site):
    alice, bob = site(*SAVE), site()
    alice.post("/api/session", json={})
    bob.post("/api/session", json={})
    state = alice.post("/api/chat", json={"message": "save AI915"}).json()
    [pending] = state["pending"]
    assert state["trips"] == []
    assert bob.post(f"/api/pending/{pending['id']}", json={"confirm": True}).status_code == 404
    done = alice.post(f"/api/pending/{pending['id']}", json={"confirm": True}).json()
    assert [t["id"] for t in done["trips"]] == ["AI915"] and done["pending"] == []
    assert done["messages"][-1]["text"].startswith("Saved. Trip: Air India AI915")
    assert bob.get("/api/state").json()["trips"] == []
    assert alice.post(f"/api/pending/{pending['id']}", json={"confirm": True}).status_code == 404


def test_access_code_gate(site, monkeypatch):
    monkeypatch.setenv("DEMO_CODE", "open-sesame")
    browser = site()
    assert browser.get("/api/state").json() == {"signed_in": False, "needs_code": True}
    assert browser.post("/api/session", json={"code": "guess"}).status_code == 403
    assert browser.post("/api/session", json={"code": "open-sesame"}).json()["signed_in"] is True


def test_daily_message_limit(site, monkeypatch):
    monkeypatch.setenv("USER_DAILY_MESSAGES", "2")
    browser = site()
    browser.post("/api/session", json={})
    codes = [browser.post("/api/chat", json={"message": "hi"}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def test_reset_clears_the_chat_but_keeps_saved_data(site):
    browser = site(*SAVE)
    browser.post("/api/session", json={})
    [pending] = browser.post("/api/chat", json={"message": "save"}).json()["pending"]
    browser.post(f"/api/pending/{pending['id']}", json={"confirm": True})
    state = browser.post("/api/reset", json={}).json()
    assert state["messages"] == [] and len(state["trips"]) == 1


def test_bad_input_is_rejected(site):
    browser = site()
    browser.post("/api/session", json={})
    assert browser.post("/api/chat", json={"message": ""}).status_code == 422
    assert browser.post("/api/chat", json={"message": "x" * 1001}).status_code == 422
    assert browser.get("/").status_code == 200
