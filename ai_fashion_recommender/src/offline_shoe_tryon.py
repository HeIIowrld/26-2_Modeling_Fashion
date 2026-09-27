"""Offline shoe experiment: crop plain studio margins and ground shoe type in metadata."""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from shoe_tryon import ShoeTryOn, PROMPT


def crop_product_reference(image):
    rgb = np.asarray(image.convert("RGB"))
    border = np.concatenate((rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]))
    background = np.median(border, axis=0)
    if np.percentile(np.max(np.abs(border.astype(float) - background), axis=1), 90) > 12:
        return image, None  # A scene/person photo is not a plain product background.
    foreground = np.max(np.abs(rgb.astype(float) - background), axis=2) > 24
    n, components, stats, _ = cv2.connectedComponentsWithStats(foreground.astype(np.uint8), 8)
    if n < 2:
        return image, None
    largest = stats[1:, cv2.CC_STAT_AREA].max()
    mask = np.isin(components, [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= largest * .10])
    if not .01 < mask.mean() < .65:
        return image, None
    ys, xs = np.where(mask)
    padding = max(8, round(max(xs.max()-xs.min(), ys.max()-ys.min()) * .08))
    box = (max(0, int(xs.min())-padding), max(0, int(ys.min())-padding),
           min(image.width, int(xs.max())+padding+1), min(image.height, int(ys.max())+padding+1))
    return image.crop(box), box


def product_prompt(product):
    types = {"로퍼": "slip-on loafers with closed low vamps",
             "더비슈즈": "leather lace-up derby shoes", "스니커즈": "sneakers",
             "러닝화": "running shoes", "메리제인": "Mary Jane shoes with instep straps",
             "부츠": "boots", "샌들": "sandals", "슬리퍼": "slides"}
    kind = types.get(product.item_type, "")
    if product.item_type == "로퍼" and any(w in product.name.lower() for w in ("태슬", "테슬", "tassel")):
        kind = "slip-on tassel loafers, each with a solid leather vamp, a strap and decorative tassels"
    return (f"The reference product is a pair of {kind}. Match that construction exactly. " if kind else "") + PROMPT


class OfflineShoeTryOn(ShoeTryOn):
    def set_product(self, product):
        self.prompt = product_prompt(product)

    def generate(self, person, reference, mask):
        cropped, box = crop_product_reference(reference)
        result = super().generate(person, cropped, mask)
        self.last_report.update(reference_crop=list(box) if box else None,
                                prompt=self.prompt, offline_reference_mode="product")
        return result
