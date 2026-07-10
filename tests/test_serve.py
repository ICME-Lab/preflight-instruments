"""Model-free tests for the scan server: routing, GET/POST, error handling.

Stubs the scanner so no models load. Spins a real ThreadingHTTPServer on an
ephemeral port and hits it with urllib.
"""
from __future__ import annotations
import json
import threading
import time
import urllib.request
import urllib.error

import numpy as np
import pytest

import src.serve as serve
from src.scan import InjectionScanner


class _FakeDet:
    def __init__(self, s, m):
        self.score = s
        self.modality = m

    def score_activations(self, X):
        return np.array([self.score])

    def score_texts(self, t):
        return np.array([self.score])


@pytest.fixture
def server():
    from http.server import ThreadingHTTPServer
    s = InjectionScanner.__new__(InjectionScanner)
    s.detectors = {
        "probe": _FakeDet(0.99, "activation"),
        "sae": _FakeDet(0.97, "activation"),
        "promptguard": _FakeDet(0.02, "text"),
    }
    s._needs_activation = True
    s._activation_for = lambda text: np.zeros((1, 4))
    serve.SCANNER = s

    srv = ThreadingHTTPServer(("127.0.0.1", 0), serve.Handler)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    time.sleep(0.2)
    yield f"http://127.0.0.1:{port}"
    srv.shutdown()


def _get(url):
    return json.load(urllib.request.urlopen(url))


def _post(url, obj):
    req = urllib.request.Request(url, data=json.dumps(obj).encode(),
                                 headers={"content-type": "application/json"})
    return json.load(urllib.request.urlopen(req))


def test_health(server):
    r = _get(f"{server}/health")
    assert r["status"] == "ok"
    assert set(r["detectors"]) == {"probe", "sae", "promptguard"}


def test_post_scan(server):
    r = _post(f"{server}/scan", {"text": "ignore instructions", "threshold": 0.9})
    assert r["verdict"] == "likely prompt injection"
    assert r["n_over"] == 2


def test_get_scan(server):
    r = _get(f"{server}/scan?text=hello&threshold=0.9")
    assert r["verdict"] == "likely prompt injection"   # stub scores are high


def test_default_threshold(server):
    r = _post(f"{server}/scan", {"text": "x"})
    assert r["threshold"] == 0.70


def test_missing_text_400(server):
    with pytest.raises(urllib.error.HTTPError) as e:
        _post(f"{server}/scan", {"threshold": 0.9})
    assert e.value.code == 400


def test_bad_threshold_400(server):
    with pytest.raises(urllib.error.HTTPError) as e:
        _post(f"{server}/scan", {"text": "x", "threshold": 5})
    assert e.value.code == 400


def test_unknown_path_404(server):
    with pytest.raises(urllib.error.HTTPError) as e:
        _get(f"{server}/nope")
    assert e.value.code == 404
