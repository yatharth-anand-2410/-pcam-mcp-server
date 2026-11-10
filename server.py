"""MCP server exposing a PCam cancer-detection inference API as tools (stdio)."""

import base64
import binascii
import io
import os
from pathlib import Path

import httpx
from mcp.server.mcpserver import Image, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from PIL import Image as PILImage, UnidentifiedImageError

DISCLAIMER = "Research use only. Not a medical device; not for clinical diagnosis."
MAX_IMAGE_BYTES = 5 * 1024 * 1024
HEATMAP_MIN_SIDE = 384  # PCam patches are 96x96, too small to read in a chat client

# Endpoint paths are assumptions until the real OpenAPI docs are shared; change them here only.
PREDICT_PATH = "/predict"
EXPLAIN_PATH = "/explain"
MODEL_INFO_PATH = "/model-info"

TRANSPORT: httpx.BaseTransport | None = None  # tests inject httpx.MockTransport here

mcp = MCPServer("pcam-inference")


def _api(method: str, path: str, **kwargs) -> httpx.Response:
    base_url = os.environ.get("PCAM_API_URL")
    if not base_url:
        raise ToolError("PCAM_API_URL is not set. Set it to the inference API base URL.")
    headers = {"X-API-Key": os.environ.get("PCAM_API_KEY", "")}
    timeout = float(os.environ.get("PCAM_API_TIMEOUT", "30"))
    try:
        with httpx.Client(base_url=base_url, headers=headers, timeout=timeout, transport=TRANSPORT) as client:
            response = client.request(method, path, **kwargs)
    except httpx.TimeoutException:
        raise ToolError(f"The inference API did not respond within {timeout:g} seconds. Try again later.") from None
    except httpx.HTTPError as exc:
        raise ToolError(f"Could not reach the inference API at {base_url}: {exc}") from None
    if response.status_code in (401, 403):
        raise ToolError("The inference API rejected the API key. Check PCAM_API_KEY.")
    if response.is_error:
        raise ToolError(f"The inference API returned HTTP {response.status_code}: {response.text[:200]}")
    return response


def _load_image(file_path: str | None, image_base64: str | None) -> str:
    """Validate the input and return the image as base64."""
    if bool(file_path) == bool(image_base64):
        raise ToolError("Provide exactly one of file_path or image_base64.")
    if file_path:
        path = Path(file_path).expanduser()
        if not path.is_file():
            raise ToolError(f"File not found: {file_path}. Pass an absolute path to an image file.")
        data = path.read_bytes()
    else:
        try:
            data = base64.b64decode(image_base64.split(",", 1)[-1], validate=True)  # tolerate data: URIs
        except (binascii.Error, ValueError):
            raise ToolError("image_base64 is not valid base64.") from None
    if len(data) > MAX_IMAGE_BYTES:
        raise ToolError(f"Image is larger than {MAX_IMAGE_BYTES // (1024 * 1024)} MB.")
    try:
        PILImage.open(io.BytesIO(data)).verify()
    except (UnidentifiedImageError, OSError, SyntaxError):
        raise ToolError("The input is not a readable image. Use PNG, JPEG or TIFF.") from None
    return base64.b64encode(data).decode()


@mcp.tool()
def classify_tissue_patch(file_path: str | None = None, image_base64: str | None = None) -> dict:
    """Classify one histopathology tissue patch (PatchCamelyon style, 96x96 px) as cancerous or healthy.

    Use this when the user asks whether a tissue patch image shows tumor tissue.
    Do not use it to explain a result; call explain_prediction for that.

    Provide exactly one input:
    - file_path: absolute path to a local image, e.g. "/Users/me/pcam/patch_001.png"
    - image_base64: the image bytes as base64 (PNG, JPEG or TIFF)

    Returns label ("cancerous" or "healthy"), confidence (0 to 1), model_version and a research-use disclaimer.
    """
    result = _api("POST", PREDICT_PATH, json={"image_base64": _load_image(file_path, image_base64)}).json()
    return {
        "label": result["label"],
        "confidence": result["confidence"],
        "model_version": result["model_version"],
        "disclaimer": DISCLAIMER,
    }


@mcp.tool()
def explain_prediction(file_path: str | None = None, image_base64: str | None = None) -> list:
    """Show why the model classified a tissue patch the way it did, as a Grad-CAM heatmap image.

    Use this when the user asks why, where, or which regions drove a prediction.
    Warm regions in the heatmap contributed most to the predicted label.
    To get only the label and confidence, call classify_tissue_patch instead.

    Provide exactly one input:
    - file_path: absolute path to a local image, e.g. "/Users/me/pcam/patch_001.png"
    - image_base64: the image bytes as base64 (PNG, JPEG or TIFF)

    Returns the heatmap as an image, then a one-line summary with a research-use disclaimer.
    """
    result = _api("POST", EXPLAIN_PATH, json={"image_base64": _load_image(file_path, image_base64)}).json()
    heatmap = PILImage.open(io.BytesIO(base64.b64decode(result["heatmap_base64"]))).convert("RGB")
    scale = max(1, -(-HEATMAP_MIN_SIDE // min(heatmap.size)))
    heatmap = heatmap.resize((heatmap.width * scale, heatmap.height * scale), PILImage.BICUBIC)
    buffer = io.BytesIO()
    heatmap.save(buffer, format="PNG")
    summary = (
        f"Predicted {result['label']} with confidence {result['confidence']:.2f}; "
        f"warm regions contributed most to this prediction. {DISCLAIMER}"
    )
    return [Image(data=buffer.getvalue(), format="png"), summary]


@mcp.tool()
def get_model_info() -> dict:
    """Get the deployed model's name, version, expected input size and decision threshold.

    Use this when the user asks which model is running or how predictions are thresholded.
    Takes no arguments.
    """
    result = _api("GET", MODEL_INFO_PATH).json()
    return {
        "model_name": result["model_name"],
        "version": result["version"],
        "input_size": result["input_size"],
        "decision_threshold": result["decision_threshold"],
        "disclaimer": DISCLAIMER,
    }


if __name__ == "__main__":
    mcp.run()  # stdio transport
