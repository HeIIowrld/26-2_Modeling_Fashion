"""Optional LIP coat/upper-clothes parser for isolated outerwear experiments.

This wrapper intentionally loads only a local ONNX checkpoint.  It does not
execute model-repository Python code or change the production FASHN parser.
The LIP taxonomy distinguishes coat (7) from upper-clothes (5), but those
labels are not guaranteed to match DeepFashion-MultiModal outer (2)/top (1).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


LIP_UPPER_CLOTHES = 5
LIP_COAT = 7
INPUT_SIDE = 473
IMAGE_MEAN = np.asarray((0.406, 0.456, 0.485), dtype=np.float32)
IMAGE_STD = np.asarray((0.225, 0.224, 0.229), dtype=np.float32)


def preprocess_schp(image: Image.Image | np.ndarray) -> np.ndarray:
    """Match the published SCHP-LIP processor's RGB resize/normalization."""
    pil = Image.fromarray(image).convert("RGB") if isinstance(image, np.ndarray) else image.convert("RGB")
    resized = pil.resize((INPUT_SIDE, INPUT_SIDE), Image.Resampling.BILINEAR)
    pixels = np.asarray(resized, dtype=np.float32) / 255.0
    pixels = (pixels - IMAGE_MEAN) / IMAGE_STD
    return np.ascontiguousarray(pixels.transpose(2, 0, 1)[None])


class OuterwearLayerParser:
    """Predict LIP labels with a separately supplied, pinned ONNX checkpoint."""

    def __init__(self, checkpoint: str | Path, *, threads: int = 4, session=None) -> None:
        self.checkpoint = Path(checkpoint)
        if session is None:
            if not self.checkpoint.is_file():
                raise FileNotFoundError(self.checkpoint)
            try:
                import onnxruntime as ort
            except ImportError as exc:
                raise RuntimeError("SCHP-LIP 실험에는 onnxruntime가 필요합니다.") from exc
            options = ort.SessionOptions()
            options.intra_op_num_threads = max(1, int(threads))
            session = ort.InferenceSession(
                str(self.checkpoint), sess_options=options,
                providers=["CPUExecutionProvider"],
            )
        self.session = session

    def predict_logits(self, image: Image.Image | np.ndarray) -> np.ndarray:
        """Return resized LIP logits as (20, height, width) for calibration."""
        pil = Image.fromarray(image).convert("RGB") if isinstance(image, np.ndarray) else image.convert("RGB")
        width, height = pil.size
        inputs = preprocess_schp(pil)
        logits = np.asarray(self.session.run(["logits"], {"pixel_values": inputs})[0])
        if logits.shape != (1, 20, INPUT_SIDE, INPUT_SIDE):
            raise RuntimeError(f"Unexpected SCHP-LIP logits shape: {logits.shape}")
        resized = cv2.resize(
            np.ascontiguousarray(logits[0].transpose(1, 2, 0)),
            (width, height), interpolation=cv2.INTER_LINEAR,
        )
        return np.ascontiguousarray(resized.transpose(2, 0, 1))

    def predict(self, image: Image.Image | np.ndarray) -> np.ndarray:
        return self.predict_logits(image).argmax(axis=0).astype(np.uint8)
