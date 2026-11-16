import asyncio
import json

import httpx
import pytest

from twt.gateway import create_app
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
