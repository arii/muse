import asyncio
import json

import httpx
import pytest

from twt.gateway import create_app
from twt.audit import report
from test_gateway import REQUEST, reply, run, setup


@pytest.mark.parametrize("origin", ["https://evil.test", "null", "http://localhost.evil.test", "http://127.0.0.1@evil.test", "http://localhost:bad"])
def test_hostile_origin_cannot_spend(store, monkeypatch, origin):
    configuration = setup(store, monkeypatch, 1)

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: pytest.fail("must not spend"))) as upstream:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(create_app(store, upstream, configuration)), base_url="http://test") as api:
                response = await api.post("/v1/chat/completions", json=REQUEST, headers={"Origin": origin})
                assert response.status_code == 403
    asyncio.run(scenario())


@pytest.mark.parametrize("content_type", [None, "text/plain", "application/x-www-form-urlencoded"])
def test_non_json_request_cannot_spend(store, monkeypatch, content_type):
    configuration = setup(store, monkeypatch, 1)

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: pytest.fail("must not spend"))) as upstream:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(create_app(store, upstream, configuration)), base_url="http://test") as api:
                response = await api.post("/v1/chat/completions", content=json.dumps(REQUEST), headers={"Content-Type": content_type} if content_type else {})
                assert response.status_code == 415
    asyncio.run(scenario())


@pytest.mark.parametrize("origin", [None, "http://localhost:8000", "http://127.0.0.1:8000", "http://[::1]:8000"])
def test_loopback_json_request_allowed(store, monkeypatch, origin):
    configuration = setup(store, monkeypatch, 1)

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=reply()))) as upstream:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(create_app(store, upstream, configuration)), base_url="http://test") as api:
                headers = {"Content-Type": "application/json; charset=utf-8"}
                if origin:
                    headers["Origin"] = origin
                assert (await api.post("/v1/chat/completions", content=json.dumps(REQUEST), headers=headers)).status_code == 200
    asyncio.run(scenario())


@pytest.mark.parametrize("first_usage,last_usage,last_status,expected,complete", [
    (True, False, 500, .05, False),
    (True, False, 200, .05, False),
    (False, True, 200, .05, False),
    (True, True, 400, .10, True),
    (True, True, 200, .10, True),
])
def test_known_cost_survives_incomplete_waterfall(store, monkeypatch, first_usage, last_usage, last_status, expected, complete):
    configuration = setup(store, monkeypatch)

    def handler(request):
        first = request.url.host == "p0.test"
        payload = reply()
        if not (first_usage if first else last_usage):
            del payload["usage"]
        return httpx.Response(500 if first else last_status, json=payload)

    _, _, headers = run(store, configuration, handler)
    trace = store.trace(headers["X-Trace-Id"])
    assert trace["known_cost_usd"] == pytest.approx(expected)
    assert trace["cost_complete"] is complete
    assert store.spend_today() == pytest.approx(expected)
    if complete:
        assert float(headers["X-Cost-Usd"]) == pytest.approx(expected)
    else:
        assert headers["X-Cost-Usd"] == "unknown" and trace["cost_usd"] is None
    monkeypatch.setenv("TWT_DAILY_BUDGET_USD", str(expected))
    assert run(store, configuration, lambda r: pytest.fail("known spend must exhaust budget"))[1] == 503


def test_stream_options_removed_from_upstream(store, monkeypatch):
    configuration = setup(store, monkeypatch, 1)
    body = {**REQUEST, "stream_options": {"include_usage": True}}

    def handler(request):
        sent = json.loads(request.content)
        assert sent["stream"] is False and "stream_options" not in sent
        return httpx.Response(200, json=reply())
    _, status, headers = run(store, configuration, handler, body)
    assert status == 200
    trace = store.trace(headers["X-Trace-Id"])
    assert trace["request"]["stream_options"] == {"include_usage": True}
    assert "stream_options" not in trace["attempts"][0]["request"]


@pytest.mark.parametrize("format_type,expected", [("text", 200), ("json_object", 503), ("json_schema", 503)])
def test_text_format_does_not_need_json_support(store, monkeypatch, format_type, expected):
    configuration = setup(store, monkeypatch, 1)
    model = store.models()[0]
    model.update(json=False, json_schema=False)
    store.replace_models("p0", [model])
    body = {**REQUEST, "response_format": {"type": format_type}}
    assert run(store, configuration, lambda r: httpx.Response(200, json=reply()), body)[1] == expected


def test_report_charges_each_attempt_provider(store, monkeypatch):
    configuration = setup(store, monkeypatch)

    def handler(request):
        if request.url.host == "p0.test":
            return httpx.Response(500, json=reply(prompt=100, completion=50))
        return httpx.Response(200, json=reply())
    run(store, configuration, handler)
    result = report(store.traces())
    rows = {row["provider"]: row for row in result["groups"]}
    assert rows["p0"]["spend_usd"] == pytest.approx(.20)
    assert rows["p1"]["spend_usd"] == pytest.approx(.05)
    assert rows["p0"]["prompt_tokens"] == 100
    assert rows["p1"]["prompt_tokens"] == 10
    assert sum(row["spend_usd"] for row in rows.values()) == pytest.approx(store.spend_today())
    assert result["flag_counts"]["FALLBACK_CASCADE"] == 1


def test_report_unknown_and_skipped_attempts(store, monkeypatch):
    configuration = setup(store, monkeypatch)
    monkeypatch.delenv("TEST_KEY_0")
    run(store, configuration, lambda r: httpx.Response(500, json={}))
    rows = report(store.traces())["groups"]
    assert len(rows) == 1 and rows[0]["provider"] == "p1"
    assert rows[0]["calls"] == 1 and rows[0]["unknown_cost_calls"] == 1
