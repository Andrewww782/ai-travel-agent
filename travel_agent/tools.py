"""What each tool actually does. Plain functions with no model and no network, so they are easy to test.

Bad input raises ToolError: MCP passes that message to the model, so it is written for the model to act on.
"""
import datetime
import json
import re
import sqlite3

from mcp.server.mcpserver.exceptions import ToolError

from .db import DATA

FLIGHTS = json.loads((DATA / "flights.json").read_text(encoding="utf-8"))
AIRPORTS = {
    "DEL": "Delhi", "BOM": "Mumbai", "BLR": "Bengaluru", "GOI": "Goa",
    "DXB": "Dubai", "SIN": "Singapore", "LHR": "London", "BKK": "Bangkok",
}  # fmt: skip
_BY_ID = {f["id"]: f for f in FLIGHTS}


def flight(flight_id) -> dict | None:
    return _BY_ID.get(str(flight_id or "").strip().upper())


def _airport(code: str) -> str:
    code = code.strip().upper()
    if code not in AIRPORTS:
        raise ToolError(f"Unknown airport code {code!r}. Supported codes: {', '.join(AIRPORTS)}.")
    return code


def _date(value: str) -> str:
    try:
        return datetime.date.fromisoformat(value.strip()).isoformat()
    except ValueError:
        raise ToolError(f"Invalid date {value!r}. Use YYYY-MM-DD.") from None


def search_flights(origin: str, destination: str, date: str) -> list[dict]:
    origin, destination, date = _airport(origin), _airport(destination), _date(date)
    return [{**f, "date": date} for f in FLIGHTS if f["origin"] == origin and f["destination"] == destination]


def search_docs(conn: sqlite3.Connection, query: str) -> list[dict]:
    # Each word is quoted so punctuation in the query cannot be read as FTS5 syntax.
    words = re.findall(r"\w+", query)[:12]
    if not words:
        raise ToolError("The query has no words to search for.")
    match = " OR ".join(f'"{w}"' for w in words)
    rows = conn.execute("SELECT title, body FROM docs WHERE docs MATCH ? ORDER BY rank LIMIT 3", (match,))
    return [dict(r) for r in rows]


def list_trips(conn: sqlite3.Connection, user_id: int) -> list[dict]:
    rows = conn.execute("SELECT flight_id, date, note FROM trips WHERE user_id = ? ORDER BY date", (user_id,))
    return [{**dict(r), **_BY_ID[r["flight_id"]]} for r in rows]


def _clean_trip(flight_id: str, date: str, note: str = "") -> dict:
    flight_id = str(flight_id).strip().upper()
    if flight_id not in _BY_ID:
        # The model cannot save a flight it made up: the id has to exist in the data.
        raise ToolError(f"No flight with id {flight_id!r}. Use an id returned by search_flights.")
    return {"flight_id": flight_id, "date": _date(str(date)), "note": str(note).strip()[:200]}


def _clean_preference(text: str) -> dict:
    text = " ".join(str(text).split())
    if not 3 <= len(text) <= 200:
        raise ToolError("A preference must be between 3 and 200 characters.")
    return {"text": text}


def save_trip(conn: sqlite3.Connection, user_id: int, flight_id: str, date: str, note: str = "") -> dict:
    trip = _clean_trip(flight_id, date, note)
    conn.execute(
        "INSERT INTO trips (user_id, flight_id, date, note) VALUES (:user_id, :flight_id, :date, :note)",
        {**trip, "user_id": user_id},
    )
    conn.commit()
    return {"saved": True, "date": trip["date"], **_BY_ID[trip["flight_id"]]}


def save_preference(conn: sqlite3.Connection, user_id: int, text: str) -> dict:
    pref = _clean_preference(text)
    conn.execute("INSERT INTO preferences (user_id, text) VALUES (?, ?)", (user_id, pref["text"]))
    conn.commit()
    return {"saved": True, "preference": pref["text"]}


def prepare_write(tool: str, args: dict) -> tuple[dict, str]:
    """Check a write before the user is asked to confirm it. Returns the cleaned arguments and the
    line shown next to the Confirm button, which is built from the arguments, never from model text.
    Raises ToolError for input the save would reject, so the model can fix it straight away."""
    if tool == "save_trip":
        trip = _clean_trip(args.get("flight_id", ""), args.get("date", ""), args.get("note", ""))
        f = _BY_ID[trip["flight_id"]]
        return trip, (
            f"Trip: {f['airline']} {f['id']}, {AIRPORTS[f['origin']]} to {AIRPORTS[f['destination']]}, "
            f"{trip['date']}, departs {f['depart']}, INR {f['price_inr']:,}"
        )
    if tool == "save_preference":
        pref = _clean_preference(args.get("text", ""))
        return pref, f"Preference: {pref['text']}"
    raise ToolError(f"{tool} cannot be confirmed by the user, so it is not available.")
