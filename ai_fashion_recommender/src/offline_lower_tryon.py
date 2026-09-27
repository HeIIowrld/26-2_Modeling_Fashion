"""Experimental reference-guided trousers adapter for isolated offline evaluation.

Uses the already installed shoe editor's pipeline. Production CatVTON defaults
are untouched. A wider editable area is permission to draw, not a target fit;
the actual result must pass preservation checks and visual review.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image, ImageOps

LOWER_PROMPT = (
    "Replace only the trousers with the exact trousers shown in the reference product image. "
    "Reproduce their color, fabric, construction, leg width, drape and hem shape. "
    "The reference garment determines the cut and silhouette, including the width at the ankles. "
    "Dress the existing person in these trousers with natural folds and lighting. "
    "Preserve the person's pose, anatomy, shirt, hands, feet, shoes and background. "
    "One realistic pair of trousers worn by the existing person."
)


def lower_edit_mask(segmentation, pose):
    import cv2

    seg = np.asarray(segmentation)
    points = getattr(pose, "landmarks", {})
    names = [f"{side}_{joint}" for side in ("left", "right") for joint in ("hip", "knee", "ankle")]
    if seg.ndim != 2 or any(name not in points or not np.isfinite(points[name]).all()
                           or points[name][2] < .5 or not all(0 <= v < 1 for v in points[name][:2]) for name in names):
        raise ValueError("Visible hips, knees and ankles are required for offline trousers editing")
    if np.isin(seg, (4, 5)).sum() > max(100, (seg == 6).sum() * .1):
        raise ValueError("The experimental trousers editor requires a trousers reference person")
    height, width = seg.shape
    hip_width = abs(points["left_hip"][0] - points["right_hip"][0]) * width
    mask = np.isin(seg, (6, 7, 14)).astype(np.uint8)
    if hip_width < 8 or (seg == 6).sum() < 100:
        raise ValueError("No reliable trousers region")
    # Add modest room along each leg; unlike a span mask this preserves most of
    # the background between separated legs. Preserve the existing shirt/feet.
    radius = round(hip_width * .65)
    for side in ("left", "right"):
        chain = [(round(points[f"{side}_{joint}"][0] * width),
                  round(points[f"{side}_{joint}"][1] * height)) for joint in ("hip", "knee", "ankle")]
        for a, b in zip(chain, chain[1:]):
            cv2.line(mask, a, b, 1, thickness=radius * 2)
    hip_y = min(points[f"{s}_hip"][1] for s in ("left", "right")) * height
    ankle_y = max(points[f"{s}_ankle"][1] for s in ("left", "right")) * height
    mask[:max(0, round(hip_y - .12 * (ankle_y - hip_y)))] = 0
    mask[min(height, round(ankle_y + .015 * height)):] = 0
    # FASHN sometimes labels skin through ripped knees as torso (16). Only
    # allow that label below the hips, inside the bounded leg editing region.
    below_hip = np.arange(height)[:, None] >= hip_y
    return mask.astype(bool) & (np.isin(seg, (0, 6, 7, 14)) | ((seg == 16) & below_hip))


class OfflineLowerTryOn:
    def __init__(self, clothing, editor, parser):
        self.clothing, self.editor, self.parser = clothing, editor, parser
        self.last_warnings, self.last_quality_reports = [], []
        self.last_render_kind = ""

    @property
    def available(self):
        return self.clothing.available and self.editor.available

    @property
    def reference_bottom_lengths(self):
        return self.clothing.reference_bottom_lengths

    def generate(self, person_image, recommendation, output_path, context=None):
        import torch
        from catvton_tryon import landmarks_to_pixels, classify_reference_bottom_length
        from shoe_tryon import edit_crop, composite_feet
        from tryon_quality import assess_tryon, load_thresholds
        from offline_render_quality import garment_family

        context = context or {}
        products = recommendation.products
        bottom = next((p for p in products if p.category == "bottom"), None)
        if bottom is None:
            result = self.clothing.generate(person_image, recommendation, output_path, context)
            self.last_quality_reports = list(self.clothing.last_quality_reports)
            self.last_warnings = list(self.clothing.last_warnings)
            return result
        if garment_family(bottom.item_type) != "pants":
            raise ValueError("Offline Flux comparison currently supports trousers only")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.last_quality_reports, self.last_warnings = [], []
        with tempfile.TemporaryDirectory(dir=output_path.parent, prefix="offline_lower_") as folder:
            tops = [p for p in products if p.category == "top"]
            source = Path(person_image)
            if tops:
                source = self.clothing.generate(source, replace(recommendation, products=tops),
                                                Path(folder) / "top.png", context)
                self.last_quality_reports = list(self.clothing.last_quality_reports)
                self.last_warnings = list(self.clothing.last_warnings)
            with Image.open(source) as opened:
                before = opened.convert("RGB")
            parsed = self.parser.parse(before, context.get("pose"))
            mask = lower_edit_mask(parsed["segmentation"], context.get("pose"))
            box = edit_crop(mask)
            crop = before.crop(box)
            scale = 1024 / max(crop.size)
            size = tuple(max(64, round(d * scale / 16) * 16) for d in crop.size)
            reference = self.clothing._prepare_garment_reference(Path(bottom.image_path), "bottom")
            pipe = self.editor._load_pipeline()
            generated = pipe(image=crop.resize(size, Image.Resampling.LANCZOS),
                image_reference=ImageOps.pad(reference, (768, 768), color="white"),
                mask_image=Image.fromarray(mask.astype(np.uint8) * 255).crop(box).resize(size, Image.Resampling.NEAREST),
                prompt=LOWER_PROMPT, height=size[1], width=size[0], strength=1.,
                num_inference_steps=4, guidance_scale=1.,
                generator=torch.Generator(device="cuda").manual_seed(self.editor.seed)).images[0]
            if generated.size != size:
                raise ValueError("Unexpected trousers edit resolution")
            result = composite_feet(before, generated, mask, box)
            after_seg = self.parser.parse(result, context.get("pose"))["segmentation"]
            ref_mask = self.clothing._reference_mask(Path(bottom.image_path).name, reference, "bottom")
            classifier = context.get("classifier")
            embed = (lambda im: classifier._encode_image(im).detach().cpu().numpy()[0]) if classifier else None
            with Image.open(bottom.image_path) as original_reference:
                reference_length = classify_reference_bottom_length(classifier, reference, product=bottom,
                    source_image=original_reference.convert("RGB")) if classifier else ""
            self.clothing.reference_bottom_lengths[bottom.product_id] = reference_length
            report = assess_tryon(category="bottom", before=np.asarray(before), after=np.asarray(result),
                edit_mask=mask, after_seg=after_seg, target_labels=(6,), before_seg=parsed["segmentation"],
                landmarks_px=landmarks_to_pixels(getattr(context.get("pose"), "landmarks", None), *before.size),
                product_name=bottom.name, reference_rgb=np.asarray(reference), reference_mask=ref_mask,
                embed=embed, reference_length=reference_length, thresholds=load_thresholds())
            self.last_quality_reports.append(report.to_dict())
            self.last_warnings.extend(report.warnings())
            self.last_warnings.append("Offline experimental trousers editor; visual review required")
            result.save(output_path)
        self.last_render_kind = "tryon"
        return output_path
