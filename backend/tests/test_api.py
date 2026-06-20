from __future__ import annotations

import io
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import app


def make_plate_image(color: tuple[int, int, int] = (180, 40, 20)) -> bytes:
    image = np.full((420, 760, 3), (235, 232, 220), dtype=np.uint8)
    cv2.rectangle(image, (190, 245), (570, 315), color, -1)
    cv2.rectangle(image, (190, 245), (570, 315), (245, 245, 245), 3)
    cv2.putText(image, "A12345", (238, 294), cv2.FONT_HERSHEY_SIMPLEX, 1.35, (245, 245, 245), 3)
    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        raise RuntimeError("failed to encode synthetic image")
    return encoded.tobytes()


class ApiTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_providers_are_available(self) -> None:
        response = self.client.get("/api/providers")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        ids = {provider["id"] for provider in payload["providers"]}
        self.assertIn("opencv_baseline", ids)
        self.assertIn("local_model", ids)
        self.assertIn("remote_server", ids)

    def test_recognize_returns_images_and_timing(self) -> None:
        image_bytes = make_plate_image()
        response = self.client.post(
            "/api/recognize",
            data={"provider": "opencv_baseline", "return_intermediate": "true"},
            files={"file": ("demo.jpg", io.BytesIO(image_bytes), "image/jpeg")},
        )
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertTrue(payload["plate_text"])
        self.assertEqual(payload["provider_used"], "opencv_baseline")
        self.assertGreater(payload["confidence"], 0)
        self.assertGreater(payload["timing_ms"]["total"], 0)
        self.assertTrue(payload["images"]["detected"].startswith("/api/outputs/"))

        image_response = self.client.get(payload["images"]["detected"])
        self.assertEqual(image_response.status_code, 200)
        self.assertEqual(image_response.headers["content-type"], "image/jpeg")

    def test_invalid_upload_is_rejected(self) -> None:
        response = self.client.post(
            "/api/recognize",
            data={"provider": "opencv_baseline", "return_intermediate": "true"},
            files={"file": ("notes.txt", io.BytesIO(b"hello"), "text/plain")},
        )
        self.assertEqual(response.status_code, 400)

    def test_local_model_provider_falls_back_cleanly(self) -> None:
        image_bytes = make_plate_image()
        env = {
            **os.environ,
            "PLATEVISION_DISABLE_TRAINED_ONNX": "1",
            "PLATEVISION_DISABLE_HYPERLPR3": "1",
        }
        with patch.dict(os.environ, env, clear=True):
            response = self.client.post(
                "/api/recognize",
                data={"provider": "local_model", "return_intermediate": "false"},
                files={"file": ("demo.jpg", io.BytesIO(image_bytes), "image/jpeg")},
            )
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["provider"], "local_model")
        self.assertEqual(payload["provider_used"], "opencv_baseline")
        self.assertTrue(any("自训练 ONNX 与 HyperLPR3" in message for message in payload["messages"]))

    def test_remote_provider_falls_back_when_disabled(self) -> None:
        self.client.post(
            "/api/settings/remote",
            json={
                "enabled": False,
                "endpoint": "http://127.0.0.1:9999/api/recognize",
                "timeout_seconds": 3,
            },
        )
        image_bytes = make_plate_image()
        response = self.client.post(
            "/api/recognize",
            data={"provider": "remote_server", "return_intermediate": "false"},
            files={"file": ("demo.jpg", io.BytesIO(image_bytes), "image/jpeg")},
        )
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["provider"], "remote_server")
        self.assertEqual(payload["provider_used"], "opencv_baseline")
        self.assertTrue(any("远程推理尚未启用" in message for message in payload["messages"]))

    def test_batch_evaluate_accepts_multiple_files(self) -> None:
        first = make_plate_image((180, 40, 20))
        second = make_plate_image((40, 160, 40))
        response = self.client.post(
            "/api/batch/evaluate",
            data={"provider": "opencv_baseline", "return_intermediate": "false"},
            files=[
                ("files", ("blue.jpg", io.BytesIO(first), "image/jpeg")),
                ("files", ("green.jpg", io.BytesIO(second), "image/jpeg")),
            ],
        )
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["metrics"]["total"], 2)
        self.assertEqual(len(payload["results"]), 2)

    def test_remote_settings_round_trip(self) -> None:
        response = self.client.post(
            "/api/settings/remote",
            json={
                "enabled": False,
                "endpoint": "http://127.0.0.1:9000/api/recognize",
                "timeout_seconds": 8,
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertFalse(payload["remote_settings"]["enabled"])
        self.assertEqual(payload["remote_settings"]["timeout_seconds"], 8)


if __name__ == "__main__":
    unittest.main()
