"""The tools as an MCP server.

The agent connects to this in-process. Run `python -m travel_agent.mcp_server` to serve the same
tools over stdio to any other MCP client.

Least privilege: read-only tools get a database connection SQLite opened read-only, and the two
write tools are marked read_only_hint=False, which is what makes the agent ask the user first.
`user_id` is filled in by the agent from the signed-in session. The model never sees that parameter.
"""
import os
from contextlib import closing

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from . import db, tools

READ = ToolAnnotations(read_only_hint=True)
WRITE = ToolAnnotations(read_only_hint=False, destructive_hint=False)


def build_server(db_path: str) -> MCPServer:
    mcp = MCPServer("travel-tools")

    @mcp.tool(annotations=READ)
    def search_flights(origin: str, destination: str, date: str) -> list[dict]:
        """Search flights between two airports on a date. Returns every matching flight with id,
        airline, departure and arrival time, stops and price in INR; an empty list means there are
        no flights on that route. Airports are IATA codes: DEL Delhi, BOM Mumbai, BLR Bengaluru,
        GOI Goa, DXB Dubai, SIN Singapore, LHR London, BKK Bangkok. Date is YYYY-MM-DD."""
        return tools.search_flights(origin, destination, date)

    @mcp.tool(annotations=READ)
    def search_docs(query: str) -> list[dict]:
        """Search the travel guide for visa rules, baggage allowances, check-in times, insurance,
        airport transfers and similar. Returns up to 3 passages. Use it for any factual travel
        question; if nothing relevant comes back, the guide does not cover it."""
        with closing(db.connect(db_path, readonly=True)) as conn:
            return tools.search_docs(conn, query)

    @mcp.tool(annotations=READ)
    def list_trips(user_id: int) -> list[dict]:
        """List the trips this user has saved, with flight details."""
        with closing(db.connect(db_path, readonly=True)) as conn:
            return tools.list_trips(conn, user_id)

    @mcp.tool(annotations=WRITE)
    def save_trip(user_id: int, flight_id: str, date: str, note: str = "") -> dict:
        """Save a flight as a planned trip for this user. flight_id must be an id returned by
        search_flights. Date is YYYY-MM-DD and must be a date the user stated: if they have not given
        one, do not call this tool, ask them for the date first. This is a plan, not a booking: nothing
        is paid."""
        with closing(db.connect(db_path)) as conn:
            return tools.save_trip(conn, user_id, flight_id, date, note)

    @mcp.tool(annotations=WRITE)
    def save_preference(user_id: int, text: str) -> dict:
        """Remember a lasting travel preference the user stated about themselves, for example
        "prefers window seats" or "no flights departing before 07:00". One preference per call."""
        with closing(db.connect(db_path)) as conn:
            return tools.save_preference(conn, user_id, text)

    return mcp


if __name__ == "__main__":
    path = os.environ.get("DB_PATH", "travel.db")
    db.init(path).close()
    build_server(path).run()
