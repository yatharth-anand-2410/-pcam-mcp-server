import base64
import io

import httpx
import pytest
from mcp.server.mcpserver import Image
from mcp.server.mcpserver.exceptions import ToolError
from PIL import Image as PILImage

import server


def png_b64(size=(96, 96)) -> str:
    buffer = io.BytesIO()
    PILImage.new("RGB", size, (180, 90, 160)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


PATCH = png_b64()
PREDICTION = {"label": "cancerous", "confidence": 0.93, "model_version": "1.2.0"}
MODEL_INFO = {"model_name": "pcam-resnet", "version": "1.2.0", "input_size": [96, 96], "decision_threshold": 0.5}


def handler(request: httpx.Request) -> httpx.Response:
    if request.headers.get("X-API-Key") != "test-key":
        return httpx.Response(401, json={"detail": "invalid API key"})
    if request.url.path == "/predict":
        return httpx.Response(200, json=PREDICTION)
    if request.url.path == "/explain":
        return httpx.Response(200, json={**PREDICTION, "heatmap_base64": PATCH})
    if request.url.path == "/model-info":
        return httpx.Response(200, json=MODEL_INFO)
    return httpx.Response(404)


@pytest.fixture(autouse=True)
def mocked_api(monkeypatch):
    monkeypatch.setenv("PCAM_API_URL", "http://inference.test")
    monkeypatch.setenv("PCAM_API_KEY", "test-key")
    monkeypatch.setattr(server, "TRANSPORT", httpx.MockTransport(handler))


def test_classify_from_base64():
    result = server.classify_tissue_patch(image_base64=PATCH)
    assert result == {**PREDICTION, "disclaimer": server.DISCLAIMER}


def test_classify_from_file_path(tmp_path):
    path = tmp_path / "patch.png"
    path.write_bytes(base64.b64decode(PATCH))
    assert server.classify_tissue_patch(file_path=str(path))["label"] == "cancerous"


def test_explain_returns_upscaled_image_and_summary():
    image, summary = server.explain_prediction(image_base64=PATCH)
    assert isinstance(image, Image)
    assert min(PILImage.open(io.BytesIO(image.data)).size) >= server.HEATMAP_MIN_SIDE
    assert "cancerous" in summary and server.DISCLAIMER in summary


def test_model_info():
    assert server.get_model_info() == {**MODEL_INFO, "disclaimer": server.DISCLAIMER}


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({}, "exactly one"),
        ({"file_path": "/a.png", "image_base64": PATCH}, "exactly one"),
        ({"file_path": "/no/such/file.png"}, "File not found"),
        ({"image_base64": "not base64!!"}, "not valid base64"),
        ({"image_base64": base64.b64encode(b"plain text").decode()}, "not a readable image"),
    ],
)
def test_invalid_inputs(kwargs, message):
    for tool in (server.classify_tissue_patch, server.explain_prediction):
        with pytest.raises(ToolError, match=message):
            tool(**kwargs)


def test_oversized_image(monkeypatch):
    monkeypatch.setattr(server, "MAX_IMAGE_BYTES", 10)
    with pytest.raises(ToolError, match="larger than"):
        server.classify_tissue_patch(image_base64=PATCH)


def test_api_timeout(monkeypatch):
    def timeout(request):
        raise httpx.ReadTimeout("timed out", request=request)

    monkeypatch.setattr(server, "TRANSPORT", httpx.MockTransport(timeout))
    for call in (lambda: server.classify_tissue_patch(image_base64=PATCH), server.get_model_info):
        with pytest.raises(ToolError, match="did not respond within"):
            call()


def test_bad_api_key(monkeypatch):
    monkeypatch.setenv("PCAM_API_KEY", "wrong")
    with pytest.raises(ToolError, match="rejected the API key"):
        server.get_model_info()


def test_api_server_error(monkeypatch):
    monkeypatch.setattr(server, "TRANSPORT", httpx.MockTransport(lambda r: httpx.Response(500, text="boom")))
    with pytest.raises(ToolError, match="HTTP 500"):
        server.get_model_info()


def test_missing_base_url(monkeypatch):
    monkeypatch.delenv("PCAM_API_URL")
    with pytest.raises(ToolError, match="PCAM_API_URL is not set"):
        server.get_model_info()
