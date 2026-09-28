"""Optional SAM 3D Body bridge for clothing-oblivious body silhouettes.

The Meta model is intentionally kept outside this repository.  This adapter
loads an approved local checkout and checkpoint only when the deployment has
them.  It reuses FITTA's already selected main-person mask as both bbox and
mask prompt, so SAM 3D Body does not need a second detector or segmentor.

The returned mask is a probabilistic editing guide, not a measurement and not
an assertion about the person's real naked body.
"""
from __future__ import annotations

from dataclasses import dataclass
import gc
import hashlib
from pathlib import Path
import sys

import cv2
import numpy as np


@dataclass(frozen=True)
class SAM3DBodyProjection:
    mask: np.ndarray
    confidence: float
    diagnostics: dict


def _largest_component(mask: np.ndarray) -> np.ndarray:
    binary = np.asarray(mask, dtype=np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if count <= 1:
        return binary.astype(bool)
    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == index


def _bbox(mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.where(mask)
    if not len(xs):
        return None
    # SAM 3D Body consumes xyxy boxes.  Keep the final pixel inside the box.
    return np.asarray([[xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]], np.float32)


class SAM3DBodyProvider:
    """Lazy, fail-closed wrapper around the official SAM 3D Body checkout."""

    def __init__(
        self,
        repo_path: str | Path,
        checkpoint_path: str | Path,
        mhr_path: str | Path,
        *,
        estimator=None,
        renderer_factory=None,
        min_visible_overlap: float = 0.55,
        release_after_inference: bool = True,
    ) -> None:
        self.repo_path = Path(repo_path).expanduser()
        self.checkpoint_path = Path(checkpoint_path).expanduser()
        self.mhr_path = Path(mhr_path).expanduser()
        self.min_visible_overlap = min_visible_overlap
        self.release_after_inference = release_after_inference
        self._estimator = estimator
        self._injected_estimator = estimator is not None
        self._renderer_factory = renderer_factory
        self._load_error = ""
        self._cache: dict[str, SAM3DBodyProjection] = {}

    @property
    def available(self) -> bool:
        if self._estimator is not None:
            return True
        return (
            (self.repo_path / "sam_3d_body").is_dir()
            and self.checkpoint_path.is_file()
            and self.mhr_path.is_file()
        )

    @property
    def load_error(self) -> str:
        return self._load_error

    def _load(self):
        if self._estimator is not None:
            return self._estimator
        if not self.available:
            raise RuntimeError("SAM 3D Body checkout/checkpoint/MHR asset is incomplete")
        try:
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("SAM 3D Body production inference requires CUDA")
            repo = str(self.repo_path.resolve())
            if repo not in sys.path:
                sys.path.insert(0, repo)
            from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
            from sam_3d_body.visualization.renderer import Renderer

            model, cfg = load_sam_3d_body(
                str(self.checkpoint_path), device=torch.device("cuda"),
                mhr_path=str(self.mhr_path),
            )
            self._estimator = SAM3DBodyEstimator(model, cfg)
            self._renderer_factory = Renderer
            return self._estimator
        except Exception as exc:
            self._load_error = f"{type(exc).__name__}: {exc}"
            raise

    def _release_estimator(self) -> None:
        if not self.release_after_inference or self._injected_estimator:
            return
        self._estimator = None
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    def estimate(self, image_rgb: np.ndarray, person_mask: np.ndarray) -> SAM3DBodyProjection | None:
        image = np.asarray(image_rgb, dtype=np.uint8)
        visible = _largest_component(np.asarray(person_mask, dtype=bool))
        if image.ndim != 3 or image.shape[:2] != visible.shape:
            return None
        cache_key = hashlib.sha256(image.tobytes() + visible.tobytes()).hexdigest()
        if cache_key in self._cache:
            return self._cache[cache_key]
        box = _bbox(visible)
        if box is None or visible.sum() < max(256, visible.size * 0.015):
            return None

        estimator = self._load()
        try:
            outputs = estimator.process_one_image(
                image,
                bboxes=box,
                masks=visible[None].astype(np.uint8),
                use_mask=True,
                inference_type="body",
            )
            if not outputs:
                return None
            output = outputs[0]
            renderer_factory = self._renderer_factory
            if renderer_factory is None:
                return None
            renderer = renderer_factory(
                focal_length=output["focal_length"], faces=estimator.faces,
            )
            canvas = np.zeros_like(image)
            rgba = renderer(
                output["pred_vertices"], output["pred_cam_t"], canvas,
                return_rgba=True, scene_bg_color=(0, 0, 0),
            )
            projected = _largest_component(np.asarray(rgba)[..., 3] > 0.20)
            area_fraction = float(projected.mean())
            if not projected.any() or not 0.02 <= area_fraction <= 0.80:
                return None

            # Clothes can extend beyond the mesh, so use precision against a gently
            # dilated visible-person mask rather than IoU as the reliability signal.
            radius = max(2, round(min(visible.shape) * 0.02))
            nearby = cv2.dilate(
                visible.astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2),
            ).astype(bool)
            visible_overlap = float(np.mean(nearby[projected]))
            if visible_overlap < self.min_visible_overlap:
                return None
            confidence = float(np.clip(0.45 + 0.5 * visible_overlap, 0.0, 0.92))
            projection = SAM3DBodyProjection(
                mask=projected,
                confidence=confidence,
                diagnostics={
                    "backend": "sam-3d-body",
                    "visible_overlap": round(visible_overlap, 4),
                    "projected_fraction": round(area_fraction, 6),
                    "bbox": [round(float(value), 2) for value in box[0]],
                    "metric_body_measurement": False,
                },
            )
            self._cache[cache_key] = projection
            while len(self._cache) > 4:
                self._cache.pop(next(iter(self._cache)))
            return projection
        finally:
            # CatVTON and FLUX need the same GPU. Release even when inference or
            # reliability validation fails; only the small projected mask is cached.
            self._release_estimator()
