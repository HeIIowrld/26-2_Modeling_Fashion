"""Reference-conditioned shoe editing after clothing VTON.

The model sees the lower-body crop and the actual product image. Only the foot
mask is composited back, so a shoe edit cannot replace the face or outfit.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from config import garment_image_path
from virtual_tryon import TryOnNotReady, VirtualTryOnAdapter

MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"
MODEL_REVISION = "e7b7dc27f91deacad38e78976d1f2b499d76a294"
PROMPT = (
    "Replace the footwear in the masked region with a matching left and right pair "
    "of the exact shoes in the reference product image. Match the reference color, "
    "material, sole and design. Fit each shoe to the existing foot direction and "
    "perspective. Keep the same two feet, ankle positions, legs, trousers and floor. "
    "Respect trouser hems overlapping the shoes. Natural contact shadows and lighting. "
    "A realistic worn pair of shoes, not a pasted product photograph. No extra feet or shoes."
)


def foot_edit_mask(segmentation, pose):
    """Fail closed for cut/hidden feet; exclude clothing and unrelated objects."""
    seg = np.asarray(segmentation)
    if seg.ndim != 2:
        raise TryOnNotReady("신발 합성에 필요한 발 영역을 찾지 못했습니다.")
    height, width = seg.shape
    points = getattr(pose, "landmarks", {})
    feet = []
    for side in ("left", "right"):
        for joint in ("ankle", "foot"):
            p = points.get(f"{side}_{joint}")
            if p is None or not np.isfinite(p).all() or p[2] < 0.5 or not (0 <= p[0] < 1 and 0 <= p[1] < 0.99):
                raise TryOnNotReady("양쪽 발목과 발끝이 보이는 사진에서 신발을 입어볼 수 있습니다.")
        feet.append((int(points[f"{side}_foot"][0] * width), int(points[f"{side}_foot"][1] * height)))
    foot_pixels = (seg == 15).astype(np.uint8)
    count, components, stats, centers = cv2.connectedComponentsWithStats(foot_pixels, 8)
    selected = np.zeros_like(foot_pixels)
    for x, y in feet:
        candidates = [(float(np.linalg.norm(centers[i] - (x, y))), i) for i in range(1, count)
                      if stats[i, cv2.CC_STAT_AREA] >= max(8, height * width * 0.00005)]
        if not candidates or min(candidates)[0] > height * 0.12:
            raise TryOnNotReady("발이 가려져 신발 영역을 안정적으로 구분하지 못했습니다.")
        selected[components == min(candidates)[1]] = 1
    radius = max(2, round(height * 0.012))
    mask = cv2.dilate(selected, np.ones((2 * radius + 1,) * 2, np.uint8)).astype(bool)
    # A small amount of leg/floor context is editable; trousers and skin elsewhere are not.
    mask &= np.isin(seg, (0, 14, 15))
    if mask.sum() < 20:
        raise TryOnNotReady("신발 합성 영역이 너무 작습니다.")
    return mask


def edit_crop(mask):
    ys, xs = np.where(mask)
    height, width = mask.shape
    margin = max(round(height * 0.10), round((xs.max() - xs.min()) * 0.2))
    return (max(0, int(xs.min()) - margin), max(0, int(ys.min()) - margin),
            min(width, int(xs.max()) + margin + 1), min(height, int(ys.max()) + margin + 1))


def composite_feet(original, generated, mask, box):
    """Feather inward only: pixels outside the mask are exactly preserved before JPEG."""
    result = np.asarray(original).copy()
    left, top, right, bottom = box
    local_mask = mask[top:bottom, left:right]
    distance = cv2.distanceTransform(local_mask.astype(np.uint8), cv2.DIST_L2, 3)
    alpha = np.clip(distance / max(2, original.height * 0.003), 0, 1)[..., None]
    patch = np.asarray(generated.resize((right-left, bottom-top), Image.Resampling.LANCZOS))
    source = result[top:bottom, left:right]
    result[top:bottom, left:right] = np.rint(source * (1-alpha) + patch * alpha).astype(np.uint8)
    return Image.fromarray(result)


class ShoeTryOn:
    def __init__(self, model_path, *, seed=42):
        self.model_path = Path(model_path)
        self.seed = seed
        self._pipeline = None
        self.last_report = {}

    @property
    def available(self):
        required = (
            "model_index.json", "transformer/diffusion_pytorch_model.safetensors",
            "transformer/config.json", "vae/diffusion_pytorch_model.safetensors",
            "vae/config.json", "text_encoder/config.json", "text_encoder/model.safetensors.index.json",
            "scheduler/scheduler_config.json", "tokenizer/tokenizer_config.json", "tokenizer/tokenizer.json")
        if not all((self.model_path / p).is_file() for p in required):
            return False
        try:
            index = json.loads((self.model_path / "text_encoder/model.safetensors.index.json").read_text())
            shards = set(index["weight_map"].values())
            return bool(shards) and all((self.model_path / "text_encoder" / p).is_file() for p in shards)
        except (OSError, ValueError, KeyError, TypeError):
            return False

    def _load_pipeline(self):
        if self._pipeline is None:
            import torch
            from diffusers import Flux2KleinInpaintPipeline
            if not self.available or not torch.cuda.is_available():
                raise TryOnNotReady("신발 합성 모델 또는 GPU가 준비되지 않았습니다.")
            self._pipeline = Flux2KleinInpaintPipeline.from_pretrained(
                str(self.model_path), torch_dtype=torch.bfloat16, local_files_only=True)
            self._pipeline.enable_model_cpu_offload()
        return self._pipeline

    def generate(self, person, reference, mask):
        import torch
        box = edit_crop(mask)
        crop = person.crop(box)
        # Avoid center cropping: retain both feet and the floor context.
        scale = 768 / max(crop.size)
        size = tuple(max(64, round(d * scale / 16) * 16) for d in crop.size)
        crop = crop.resize(size, Image.Resampling.LANCZOS)
        mask_image = Image.fromarray(mask.astype(np.uint8) * 255).crop(box).resize(size, Image.Resampling.NEAREST)
        product = ImageOps.pad(reference.convert("RGB"), (512, 512), color="white")
        pipe = self._load_pipeline()
        output = pipe(image=crop, image_reference=product, mask_image=mask_image,
                      prompt=PROMPT, height=size[1], width=size[0], strength=1.0,
                      num_inference_steps=4, guidance_scale=1.0,
                      generator=torch.Generator(device="cuda").manual_seed(self.seed)).images[0]
        if output.size != size:
            raise RuntimeError("신발 생성 결과 해상도가 요청과 다릅니다.")
        result = composite_feet(person, output, mask, box)
        changed = np.max(np.abs(np.asarray(result).astype(float) - np.asarray(person).astype(float)), axis=2)
        changed_fraction = float(np.mean(changed[mask] > 8))
        if changed_fraction < 0.02:
            raise TryOnNotReady("신발이 충분히 변경되지 않았습니다. 다른 상품 사진으로 다시 시도해 주세요.")
        self.last_report = {"backend": MODEL_ID, "revision": MODEL_REVISION,
                            "changed_fraction": changed_fraction, "crop": list(box),
                            "outside_mask_preserved": bool(np.all(changed[~mask] == 0)),
                            "reference_fidelity": "not_calibrated"}
        return result


class OutfitTryOn(VirtualTryOnAdapter):
    """CatVTON for clothes, reference-guided local inpainting for shoes."""
    def __init__(self, clothing, shoes, parser=None):
        super().__init__(enabled=clothing.available)
        self.clothing, self.shoes, self.parser = clothing, shoes, parser
        self.last_warnings, self.last_quality_reports = [], []
        self.last_render_kind = ""

    @property
    def supported_categories(self):
        return {"top", "bottom", "shoes"} if self.shoes.available else {"top", "bottom"}

    @property
    def reference_bottom_lengths(self):
        return self.clothing.reference_bottom_lengths

    def generate(self, person_image, recommendation, output_path, context=None):
        products = recommendation.products
        categories = [p.category for p in products]
        if len(categories) != len(set(categories)) or set(categories) - self.supported_categories:
            raise TryOnNotReady("지원하는 카테고리별 상품을 하나씩 선택해 주세요.")
        self.last_warnings, self.last_quality_reports, self.last_render_kind = [], [], ""
        footwear = next((p for p in products if p.category == "shoes"), None)
        if footwear is None:
            result = self.clothing.generate(person_image, recommendation, output_path, context)
            self.last_warnings = list(self.clothing.last_warnings)
            self.last_quality_reports = list(self.clothing.last_quality_reports)
            self.last_render_kind = self.clothing.last_render_kind
            return result
        context = context or {}
        # Reject hidden feet before spending time on clothing generation.
        foot_edit_mask(context.get("segmentation"), context.get("pose"))
        path = Path(footwear.image_path).expanduser() if footwear.image_path else None
        if path is not None and not path.is_absolute():
            path = garment_image_path(footwear.image_path)
        if path is None or not path.is_file():
            raise TryOnNotReady("추천 신발의 실제 상품 이미지가 필요합니다.")
        if self.parser is None:
            from clothing_parser import ClothingParser
            self.parser = ClothingParser(use_fashn=True)
        if self.parser.backend != "fashn-human-parser":
            raise TryOnNotReady("신발 합성에 필요한 의류 파서를 사용할 수 없습니다.")
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        # Never put a clothing-only intermediate at the final cache filename.
        with tempfile.TemporaryDirectory(prefix="shoe_stage_", dir=output.parent) as folder:
            source = Path(person_image)
            clothes = [p for p in products if p.category != "shoes"]
            if clothes:
                from dataclasses import replace
                source = self.clothing.generate(person_image, replace(recommendation, products=clothes),
                                                Path(folder) / "clothes.png", context)
                self.last_warnings = list(self.clothing.last_warnings)
                self.last_quality_reports = list(self.clothing.last_quality_reports)
            with Image.open(source) as opened:
                person = opened.convert("RGB")
            parsed = self.parser.parse(np.asarray(person), context.get("pose"))
            mask = foot_edit_mask(parsed["segmentation"], context.get("pose"))
            with Image.open(path) as opened:
                result = self.shoes.generate(person, opened.convert("RGB"), mask)
            self.last_quality_reports.append(dict(self.shoes.last_report))
            self.last_warnings.append("신발 합성은 실험 기능입니다. 양쪽 신발의 형태와 상품 반영을 확인해 주세요.")
            staged = Path(folder) / ("result" + output.suffix)
            result.save(staged, quality=95)
            os.replace(staged, output)
        self.last_render_kind = "tryon"
        return output
