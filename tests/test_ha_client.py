import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from app.ha_client import HAClient


class MockHAHandler(BaseHTTPRequestHandler):
    states = {
        "sensor.test": {"state": "1", "attributes": {"unit": "W"}},
    }
    last_headers = None

    def _send_json(self, payload, code=200):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):  # noqa: N802
        MockHAHandler.last_headers = self.headers
        path = self.path
        if path.startswith("/api/states/"):
            entity_id = path.split("/api/states/", 1)[1]
            payload = MockHAHandler.states.get(entity_id, {"state": "unknown"})
            self._send_json(payload)
        else:
            self._send_json({"error": "not found"}, code=404)

    def do_POST(self):  # noqa: N802
        MockHAHandler.last_headers = self.headers
        path = self.path
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length else b"{}"
        data = json.loads(body.decode("utf-8"))
        if path.startswith("/api/states/"):
            entity_id = path.split("/api/states/", 1)[1]
            state = data.get("state")
            attributes = data.get("attributes") or {}
            MockHAHandler.states[entity_id] = {"state": state, "attributes": attributes}
            self._send_json(MockHAHandler.states[entity_id])
        else:
            self._send_json({"error": "not found"}, code=404)

    def log_message(self, *args, **kwargs):  # noqa: D401
        """Silence base handler logging."""
        return


@pytest.fixture
def ha_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), MockHAHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    base_url = f"http://{host}:{port}"
    yield base_url
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


def test_get_state_and_headers(ha_server):
    client = HAClient(ha_server, token="abcdefghijkl")
    state = client.get_state("sensor.test")
    assert state["state"] == "1"
    assert MockHAHandler.last_headers.get("Authorization") == "Bearer abcdefghijkl"
    headers = client._headers()
    assert headers["Authorization"].endswith("ijkl")


def test_set_state_updates_server(ha_server):
    client = HAClient(ha_server, token="token1234")
    client.set_state("sensor.new", "42", {"foo": "bar"})
    updated = MockHAHandler.states["sensor.new"]
    assert updated["state"] == "42"
    assert updated["attributes"]["foo"] == "bar"


def test_log_token_masking(caplog):
    caplog.set_level("INFO")
    HAClient("http://example.com", token="abcdefghijkl")
    assert any("abcd****ijkl" in rec.message for rec in caplog.records)


def test_log_token_short_masks_all(caplog):
    caplog.set_level("INFO")
    HAClient("http://example.com", token="12345678")
    assert any("********" in rec.message for rec in caplog.records)


def test_log_token_missing_warns(caplog):
    caplog.set_level("WARNING")
    HAClient("http://example.com", token="")
    assert any("HA token is not set" in rec.message for rec in caplog.records)
