"""Tests for the viewer: the catalogue, the 'before' labels, and the HTTP server.

    python -m pytest tests -q
"""

from __future__ import annotations

import json
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))

import catalogue  # noqa: E402
import server  # noqa: E402


def _results(task):
    return json.loads((ROOT / task["folder"] / "results.json").read_text(encoding="utf-8"))


def test_every_task_has_files_on_disk():
    for t in catalogue.TASKS:
        items = catalogue.samples(t)
        assert len(items) >= 3, t["id"]
        for item in items:
            for key, val in item.items():
                if key in ("input", "before", "after", "truth", "bicubic", "grid"):
                    assert (ROOT / val).is_file(), (t["id"], val)


def test_image_tasks_show_four_subjects():
    for t in catalogue.TASKS:
        if t["kind"] in ("restore", "sr", "prompt"):
            assert len(catalogue.samples(t)) == 4, t["id"]


def test_before_label_matches_the_run_history():
    """'Earlier checkpoint' exactly when the saved before-score sits inside the run's own
    history (the run resumed); 'Before fine-tuning' only when it is below all of it."""
    for t in catalogue.TASKS:
        if t["kind"] not in ("restore", "sr"):
            continue
        res = _results(t)
        before = res["before_val"]["psnr"]
        hist = [h.get("psnr", h.get("val_psnr")) for h in res.get("history", [])]
        hist = [h for h in hist if isinstance(h, (int, float))]
        resumed = any(abs(h - before) < 0.3 for h in hist) or before > min(hist)
        if t["before"] == catalogue.EARLIER:
            assert resumed, t["id"]
        else:
            assert before < min(hist), t["id"]


def test_text_to_image_history_starts_mid_run():
    t = catalogue.BY_ID["text-to-image"]
    assert t["before"] == catalogue.EARLIER
    assert _results(t)["history"][0]["step"] > 0


def test_sample_scores_are_computed_from_the_pngs():
    s = catalogue.sample_scores("document-shadow-removal", "71-5")
    assert set(s) == {"input", "before", "after"}
    assert s["input"] < s["before"] < s["after"]
    sr = catalogue.sample_scores("sr-esrgan", "0012")
    assert "bicubic" in sr and sr["after"] > sr["bicubic"]


def test_unknown_sample_scores_empty():
    assert catalogue.sample_scores("cloud-removal", "nope") == {}


def test_split_scores_read_results_json():
    rows = catalogue.split_scores(catalogue.BY_ID["cloud-removal"])
    test_row = [r for r in rows if r["split"] == "test"][0]
    assert test_row["value"] == pytest.approx(23.6964, abs=1e-4)
    cyc = catalogue.split_scores(catalogue.BY_ID["cyclegan-s2w"])
    assert cyc[0]["metric"] == "cycle-consistency L1" and cyc[0]["value"] > 0


@pytest.mark.parametrize("bad", ["", "../README.md", "app/server.py", "/etc/passwd",
                                 "cloud-removal/results.json", "cloud-removal/samples/../train_cloud.py",
                                 "cloud-removal\\samples\\00355_input_cloudy.png"])
def test_public_file_refuses_anything_outside_samples(bad):
    assert catalogue.public_file(bad) is None


def test_public_file_allows_a_sample():
    assert catalogue.public_file("cloud-removal/samples/00355_input_cloudy.png") is not None


@pytest.fixture(scope="module")
def base_url():
    srv = server.make_server(0)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def _get(url):
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            return r.status, r.headers.get("Content-Type", ""), r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("Content-Type", ""), e.read()


def test_server_routes(base_url):
    code, ctype, body = _get(base_url + "/")
    assert code == 200 and "text/html" in ctype and b"Generative Vision Lab" in body
    for asset in ("/static/app.js", "/static/style.css"):
        assert _get(base_url + asset)[0] == 200
    code, _, body = _get(base_url + "/api/catalogue")
    cat = json.loads(body)
    assert code == 200 and len(cat["tasks"]) == len(catalogue.TASKS)
    first = cat["tasks"][0]["samples"][0]["input"]
    code, ctype, body = _get(base_url + "/file?p=" + first)
    assert code == 200 and ctype == "image/png" and body[:4] == b"\x89PNG"
    code, ctype, body = _get(base_url + "/thumb?p=super-resolution/esrgan/samples/0012_input_lr.png")
    assert code == 200 and ctype == "image/jpeg" and len(body) < 60_000
    code, _, body = _get(base_url + "/api/scores?task=cloud-removal&sample=00355")
    assert code == 200 and "after" in json.loads(body)


def test_server_errors_are_json_404(base_url):
    for path in ("/nope", "/file?p=../README.md", "/thumb?p=cloud-removal/results.json", "/api/scores?task=x&sample=y",
                 "/api/scores?task=cloud-removal&sample=zzz", "/static/../server.py"):
        code, ctype, _ = _get(base_url + path)
        assert code == 404 and "json" in ctype, path
