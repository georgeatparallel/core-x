"""Parallel HTTP retry regressions; no vendor credentials or remote services needed."""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from unittest.mock import patch

import pytest
import requests

from core import parallel_client as client


def response(status, *, retry_after=None, body=None):
    result = requests.Response()
    result.status_code = status
    result._content = json.dumps(body or {"error": "try later"}).encode()
    if retry_after is not None:
        result.headers["Retry-After"] = retry_after
    return result


@pytest.mark.parametrize("status", [429, 503])
@pytest.mark.parametrize("header,delay", [
    ("12", 12.0), ("0", 0.0), ("1.5", 1.5),
    ("Wed, 21 Oct 2015 07:28:00 GMT", 10.0),
    ("Wednesday, 21-Oct-15 07:28:00 GMT", 10.0),
    ("Wed Oct 21 07:28:00 2015", 10.0),
    ("Wed, 21 Oct 2015 07:27:00 GMT", 0.0),
    (None, 1.25), ("", 1.25), ("invalid", 1.25), ("-1", 1.25),
    ("NaN", 1.25), ("Infinity", 1.25), ("1e999", 1.25),
])
def test_search_retries_transient_response(monkeypatch, status, header, delay):
    monkeypatch.setenv("PARALLEL_API_KEY", "test-key")
    excerpts = {"results": [{"url": "https://example.com", "excerpts": ["Company profile"]}]}
    with patch.object(client.requests, "request", side_effect=[
        response(status, retry_after=header), response(200, body=excerpts),
    ]) as send, patch.object(client.time, "sleep") as sleep, \
            patch.object(client.time, "time", return_value=1445412470), \
            patch.object(client.random, "random", return_value=0.25):
        assert client.search(objective="Find company information") == excerpts
    sleep.assert_called_once_with(delay)
    assert send.call_count == 2
    for call in send.call_args_list:
        assert call.args == ("POST", "https://api.parallel.ai/v1/search")
        assert call.kwargs["headers"]["x-api-key"] == "test-key"
        assert call.kwargs["headers"]["User-Agent"] == "core-x (https://github.com/bencrane/core-x)"
        assert call.kwargs["json"] == {"mode": "advanced", "objective": "Find company information"}
        assert call.kwargs["timeout"] == (30, 180)


def test_retry_exhaustion_keeps_budget_and_backoff(monkeypatch):
    monkeypatch.setenv("PARALLEL_API_KEY", "test-key")
    with patch.object(client.requests, "request", return_value=response(503, retry_after="bad")) as send, \
            patch.object(client.time, "sleep") as sleep, \
            patch.object(client.random, "random", return_value=0.25):
        with pytest.raises(client.ParallelError, match="failed after 6 attempts"):
            client.search(objective="Find company information")
    assert send.call_count == 6
    assert [call.args[0] for call in sleep.call_args_list] == [1.25, 2.25, 4.25, 8.25, 16.25]


@pytest.mark.parametrize("status", [400, 401, 403])
def test_terminal_errors_do_not_retry(monkeypatch, status):
    monkeypatch.setenv("PARALLEL_API_KEY", "test-key")
    with patch.object(client.requests, "request", return_value=response(status, retry_after="bad")) as send, \
            patch.object(client.time, "sleep") as sleep:
        with pytest.raises(client.ParallelError) as error:
            client.search(objective="Find company information")
    assert error.value.status == status
    assert send.call_count == 1
    sleep.assert_not_called()


def test_task_result_preserves_408_and_blocking_timeout(monkeypatch):
    monkeypatch.setenv("PARALLEL_API_KEY", "test-key")
    with patch.object(client.requests, "request", return_value=response(408)) as send, \
            patch.object(client.time, "sleep") as sleep:
        assert client.get_task_result("run_test", timeout=600) == {"_status": 408}
    assert send.call_args.args == ("GET", "https://api.parallel.ai/v1/tasks/runs/run_test/result")
    assert send.call_args.kwargs["params"] == {"timeout": 600}
    assert send.call_args.kwargs["timeout"] == (30, 630)
    sleep.assert_not_called()


def test_transport_timeout_retries_and_interrupt_propagates(monkeypatch):
    monkeypatch.setenv("PARALLEL_API_KEY", "test-key")
    with patch.object(client.requests, "request", side_effect=[requests.Timeout("read timed out"), response(200)]), \
            patch.object(client.time, "sleep") as sleep:
        client.search(objective="Find company information")
    assert sleep.call_count == 1
    with patch.object(client.requests, "request", side_effect=KeyboardInterrupt), \
            patch.object(client.time, "sleep") as sleep:
        with pytest.raises(KeyboardInterrupt):
            client.search(objective="Find company information")
    sleep.assert_not_called()


def test_search_http_roundtrip(monkeypatch):
    """Exercise real requests serialization, retry and result consumption at the caller."""
    observed = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            observed.append((self.path, self.headers, json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
            self.send_response(429 if len(observed) == 1 else 200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Retry-After", "Wed, 21 Oct 2015 07:28:00 GMT")
            self.end_headers()
            self.wfile.write(json.dumps({"results": [{
                "url": "https://example.com/company", "excerpts": ["Aerospace supplier"],
            }]}).encode())

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    monkeypatch.setattr(client, "PARALLEL_BASE", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setenv("PARALLEL_API_KEY", "local-fixture-key")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    thread.start()
    try:
        result = client.search(search_queries=["aerospace supplier"])
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert result["results"][0]["excerpts"] == ["Aerospace supplier"]
    assert len(observed) == 2
    for path, headers, body in observed:
        assert path == "/v1/search"
        assert headers["User-Agent"] == "core-x (https://github.com/bencrane/core-x)"
        assert headers["x-api-key"] == "local-fixture-key"
        assert body == {"mode": "advanced", "search_queries": ["aerospace supplier"]}
