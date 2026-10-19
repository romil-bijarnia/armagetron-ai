"""The default training UI is an observer of one live game, with no playback controls."""

import json
from urllib.request import urlopen

import torch

from arma_ai.brain import Hub, Metrics, serve


def test_training_entry_serves_live_view_and_assets():
    hub = Hub(1, source="training")
    server, port = serve(hub, 0)
    try:
        with urlopen(f"http://127.0.0.1:{port}/") as r:
            html = r.read().decode()
        assert "Live game" in html
        assert "Hyperparameters & resources" in html
        assert 'id="player-select"' in html and 'id="sensor-map"' in html
        for obsolete in ('id="history"', 'id="pause"', 'id="step"', 'id="speed"', 'Policy inspection'):
            assert obsolete not in html
        for path in ("live.js", "live.css", "arena3d.js", "render.js", "architecture3d.js"):
            with urlopen(f"http://127.0.0.1:{port}/{path}") as r:
                assert r.status == 200 and len(r.read()) > 100
    finally:
        server.shutdown()
        server.server_close()


def test_inspection_entry_names_blacklight_and_keeps_architecture_version_separate():
    server, port = serve(Hub(1, source="inspection"), 0)
    try:
        with urlopen(f"http://127.0.0.1:{port}/") as response:
            html = response.read().decode()
        assert "<title>Blacklight · Armagetron</title>" in html
        assert "<h1>Blacklight</h1>" in html
        with urlopen(f"http://127.0.0.1:{port}/brain.js") as response:
            script = response.read().decode()
        assert "V${t.version} architecture" in script
    finally:
        server.shutdown()
        server.server_close()


def test_metrics_use_saved_configuration_then_actual_running_values(tmp_path):
    torch.save({"config": {"lr": .0002, "games": 256}}, tmp_path / "latest.pt")
    path = tmp_path / "metrics.jsonl"
    path.write_text(json.dumps({"update": 12, "time": 100, "pg": .5}) + "\n")
    metrics = Metrics(path)
    assert metrics.poll()
    saved = json.loads(metrics.payload())["configuration"]
    assert saved == {"values": {"lr": .0002, "games": 256}, "source": "saved checkpoint"}
    (tmp_path / "progress.json").write_text(json.dumps({"config": {"mode": "background", "lr": .0001, "games": 96}, "time": 101}))
    actual = json.loads(metrics.payload())["configuration"]
    assert actual["source"] == "trainer" and actual["values"]["games"] == 96
    assert actual["values"]["lr"] == .0001
