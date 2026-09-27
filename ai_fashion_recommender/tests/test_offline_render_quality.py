from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from offline_render_quality import assess_preservation, pants_geometry, reference_rank
from offline_lower_tryon import lower_edit_mask


def trousers(taper=False, gap=20):
    seg = np.zeros((250, 240), dtype=np.uint8)
    for y in range(25, 225):
        width = round(40 * (1 - .65 * (y - 25) / 200)) if taper else 40
        seg[y, 80-width:80] = 6
        seg[y, 80+gap:80+gap+width] = 6
    return seg


class PreservationTests(unittest.TestCase):
    products = [{"category": "top", "item_type": "티셔츠"}, {"category": "bottom", "item_type": "치노 팬츠"}]
    observed = {"input_valid": True, "upper_type": "티셔츠", "lower_type": "팬츠",
                "lower_subtype": "치노 팬츠", "layering_state": "단일 상의"}
    reports = [{"category": c, "checks": [{"passed": True}]} for c in ("top", "bottom")] + [
        {"backend": "shoes", "outside_mask_preserved": True}]

    def test_wide_trousers_becoming_tapered_are_not_admitted(self):
        result = assess_preservation(self.products, self.observed, trousers(True), pants_geometry(trousers()), self.reports)
        self.assertEqual(result["status"], "quality_failed")
        self.assertIn("pants_silhouette", result["failed"])

    def test_distance_between_legs_does_not_change_shape_measure(self):
        self.assertEqual(pants_geometry(trousers(gap=5))["hem_thigh"],
                         pants_geometry(trousers(gap=90))["hem_thigh"])

    def test_harem_conflict_rejected_even_when_fit_geometry_matches(self):
        result = assess_preservation(self.products, {**self.observed, "lower_subtype": "하렘 팬츠"},
                                     trousers(), pants_geometry(trousers()), self.reports)
        self.assertIn("bottom_subtype_conflict", result["failed"])

    def test_unknown_reference_shape_requires_assessment(self):
        result = assess_preservation(self.products, self.observed, trousers(), {"measurable": False}, self.reports)
        self.assertEqual(result["status"], "unassessed")

    def test_automatic_pass_still_requires_visual_review(self):
        result = assess_preservation(self.products, self.observed, trousers(), pants_geometry(trousers()), self.reports)
        self.assertEqual(result["status"], "auto_passed")
        self.assertTrue(result["visual_review_required"])

    def test_clipped_hem_and_stray_shoe_label_do_not_create_false_measurement(self):
        original = trousers()
        noisy = original.copy()
        noisy[-3:, 30:33] = 6
        self.assertEqual(pants_geometry(noisy), pants_geometry(original))
        cropped = original[:180]
        self.assertFalse(pants_geometry(cropped)["measurable"])

    def test_reference_selection_rejects_dress_and_incompatible_cuffs(self):
        reference = {"pose": {"valid": True, "body_shape_confidence": .9, "full_body_score": .99},
                     "outfit": {**self.observed, "bottom_length": "긴바지"}, "feet_error": ""}
        self.assertTrue(reference_rank(reference, self.products, pants_geometry(trousers()), trousers())["eligible"])
        self.assertFalse(reference_rank(reference, self.products, pants_geometry(trousers()), trousers(True))["eligible"])
        dress = trousers().copy()
        dress[dress == 6] = 4
        self.assertIn("dress_or_skirt_overlap", reference_rank(reference, self.products, pants_geometry(trousers()), dress)["reasons"])

    def test_lower_editor_protects_top_hands_and_feet(self):
        seg = trousers()
        seg[5:40, 40:160] = 3
        seg[230:, 40:160] = 15
        seg[140:155, 80:100] = 13
        landmarks = {f"{s}_{j}": (x, y, 1.) for s, x in (("left", .30), ("right", .50))
                     for j, y in (("hip", .18), ("knee", .50), ("ankle", .88))}
        mask = lower_edit_mask(seg, SimpleNamespace(landmarks=landmarks))
        self.assertFalse(mask[np.isin(seg, (3, 13, 15))].any())
        self.assertTrue(mask[seg == 6].any())
        landmarks["left_ankle"] = (.3, .88, .1)
        with self.assertRaises(ValueError):
            lower_edit_mask(seg, SimpleNamespace(landmarks=landmarks))


if __name__ == "__main__":
    unittest.main()
