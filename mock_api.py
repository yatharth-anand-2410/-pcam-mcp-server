"""Stand-in for the real inference API, for local demos only. Predictions are fake.

Run: uv run mock_api.py   (serves on http://127.0.0.1:8765, API key "dev-key")
"""

import base64
import io
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

from PIL import Image, ImageFilter, ImageOps

API_KEY = "dev-key"
MODEL = {"model_name": "mock-pcam-resnet", "version": "0.0-mock", "input_size": [96, 96], "decision_threshold": 0.5}


def predict(image: Image.Image) -> dict:
    # Fake score from mean darkness, so different images give different answers.
    score = 1 - sum(image.convert("L").resize((8, 8)).getdata()) / (64 * 255)
    label = "cancerous" if score >= MODEL["decision_threshold"] else "healthy"
    confidence = score if label == "cancerous" else 1 - score
    return {"label": label, "confidence": round(confidence, 4), "model_version": MODEL["version"]}


def heatmap(image: Image.Image) -> str:
    gray = ImageOps.invert(image.convert("L")).filter(ImageFilter.GaussianBlur(6))
    overlay = Image.blend(image.convert("RGB"), ImageOps.colorize(gray, "navy", "red", mid="yellow"), 0.5)
    buffer = io.BytesIO()
    overlay.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


class Handler(BaseHTTPRequestHandler):
    def _send(self, status: int, body: dict):
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _authorized(self) -> bool:
        if self.headers.get("X-API-Key") == API_KEY:
            return True
        self._send(401, {"detail": "invalid API key"})
        return False

    def do_GET(self):
        if self._authorized():
            self._send(200, MODEL) if self.path == "/model-info" else self._send(404, {"detail": "not found"})

    def do_POST(self):
        if not self._authorized():
            return
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        image = Image.open(io.BytesIO(base64.b64decode(body["image_base64"])))
        if self.path == "/predict":
            self._send(200, predict(image))
        elif self.path == "/explain":
            self._send(200, {**predict(image), "heatmap_base64": heatmap(image)})
        else:
            self._send(404, {"detail": "not found"})


if __name__ == "__main__":
    print("Mock inference API on http://127.0.0.1:8765 (API key: dev-key)")
    HTTPServer(("127.0.0.1", 8765), Handler).serve_forever()
