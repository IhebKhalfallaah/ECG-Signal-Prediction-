import io
import os
from pathlib import Path

import pytest
from flask import template_rendered

from app import create_app
from tests.conftest import ROOT

SAMPLE_DIR = ROOT / "data" / "sample"


@pytest.fixture
def app(tmp_path):
    return create_app({
        "TESTING": True,
        "UPLOAD_FOLDER": str(tmp_path / "uploads"),
        "PLOTS_DIR": str(tmp_path / "plots"),
    })


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def rendered(app):
    """Templates rendered during a request, as (template_name, context) pairs."""
    captured = []

    def record(sender, template, context, **extra):
        captured.append((template.name, context))

    template_rendered.connect(record, app)
    yield captured
    template_rendered.disconnect(record, app)


def _upload(client, files):
    data = {"name": "Jane", "ecg_files": [(io.BytesIO(content), name) for name, content in files]}
    return client.post("/Predict", data=data, content_type="multipart/form-data")


def test_home_ok(client):
    assert client.get("/").status_code == 200


def test_login_success_redirects(client):
    response = client.post("/login", data={"username": "admin", "password": "1234"})
    assert response.status_code == 302
    assert response.headers["Location"].endswith("/predictPage")


def test_login_wrong_password(client):
    response = client.post("/login", data={"username": "admin", "password": "nope"})
    assert b"Invalid username or password" in response.data


def test_login_reads_env(monkeypatch, tmp_path):
    monkeypatch.setenv("ECG_PASSWORD", "secret")
    client = create_app({"TESTING": True}).test_client()
    assert client.post("/login", data={"username": "admin", "password": "secret"}).status_code == 302
    assert client.post("/login", data={"username": "admin", "password": "1234"}).status_code == 200


# Note: predict.html does not display `error` (original behavior), so assert on the template context.
def test_predict_missing_file(client, rendered):
    _upload(client, [("e0103.dat", b"x"), ("e0103.hea", b"x")])
    name, context = rendered[-1]
    assert name == "predict.html"
    assert "Missing .dat, .hea, or .xyz file" in context["error"]
    assert context["name"] == "Jane"


def test_predict_two_records(client, rendered):
    _upload(client, [("a.dat", b"x"), ("b.hea", b"x")])
    name, context = rendered[-1]
    assert name == "predict.html"
    assert "exactly one ECG record" in context["error"]


def test_predict_sample_end_to_end(client, app, golden, monkeypatch):
    monkeypatch.chdir("/tmp")
    files = [(p.name, p.read_bytes()) for p in sorted(SAMPLE_DIR.glob("e0103.*"))]
    response = _upload(client, files)
    assert response.status_code == 200
    assert f"Prediction: {golden['prediction'].capitalize()}" in response.get_data(as_text=True)
    assert len(list(Path(app.config["PLOTS_DIR"]).glob("e0103_*.png"))) == 4


def test_model_not_in_static(client):
    assert client.get("/static/svm_model_C60_gamma001.pkl").status_code == 404
