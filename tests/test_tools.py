import pytest
from mcp.server.mcpserver.exceptions import ToolError

from travel_agent import tools

from .helpers import run


def test_search_flights_filters_by_route_and_accepts_lower_case():
    found = tools.search_flights("del", " dxb ", "2027-03-10")
    assert {f["id"] for f in found} == {"AI915", "EK511", "6E1461"}
    assert all(f["date"] == "2027-03-10" for f in found)


def test_a_route_with_no_flights_is_an_empty_list_not_an_error():
    assert tools.search_flights("GOI", "LHR", "2027-03-10") == []


@pytest.mark.parametrize("args", [("XXX", "DXB", "2027-03-10"), ("DEL", "DXB", "10 March"), ("DEL", "DXB", "2027-02-30")])
def test_bad_search_input_gives_a_message_the_model_can_act_on(args):
    with pytest.raises(ToolError, match="Unknown airport code|Invalid date"):
        tools.search_flights(*args)


def test_search_docs_finds_the_right_passage():
    async def body(env):
        assert "UAE" in tools.search_docs(env.conn, "Do I need a visa for Dubai?")[0]["title"]
        assert "IndiGo" in tools.search_docs(env.conn, "indigo baggage allowance to singapore")[0]["title"]
        assert len(tools.search_docs(env.conn, "visa")) == 3

    run(body)


def test_search_docs_treats_search_syntax_as_plain_words():
    async def body(env):
        tools.search_docs(env.conn, 'visa" OR NOT (title: *')  # must not raise
        with pytest.raises(ToolError):
            tools.search_docs(env.conn, "?!")

    run(body)


def test_prepare_write_cleans_arguments_and_builds_the_confirmation_line():
    args, summary = tools.prepare_write("save_trip", {"flight_id": " ai915 ", "date": "2027-03-10"})
    assert args == {"flight_id": "AI915", "date": "2027-03-10", "note": ""}
    assert summary == "Trip: Air India AI915, Delhi to Dubai, 2027-03-10, departs 08:10, INR 18,500"
    assert tools.prepare_write("save_preference", {"text": "  window \n seat "}) == (
        {"text": "window seat"},
        "Preference: window seat",
    )


@pytest.mark.parametrize(
    "tool, args",
    [
        ("save_trip", {"flight_id": "ZZ999", "date": "2027-03-10"}),
        ("save_trip", {"flight_id": "AI915", "date": "soon"}),
        ("save_trip", {}),
        ("save_preference", {"text": "x"}),
        ("save_preference", {"text": "y" * 201}),
        ("delete_everything", {}),
    ],
)
def test_prepare_write_rejects_what_the_save_would_reject(tool, args):
    with pytest.raises(ToolError):
        tools.prepare_write(tool, args)
