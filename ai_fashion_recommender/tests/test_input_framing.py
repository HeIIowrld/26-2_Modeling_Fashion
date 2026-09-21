import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from quality_checker import QualityChecker


def check(point=(0.5, 0.9, 0.99)):
    names = ("left_ankle", "right_ankle", "left_foot", "right_foot")
    landmarks = {
        "left_shoulder": (0.35, 0.25, 0.99),
        "right_shoulder": (0.65, 0.25, 0.99),
        "left_hip": (0.42, 0.55, 0.99),
        "right_hip": (0.58, 0.55, 0.99),
        **{name: (0.5, 0.9, 0.99) for name in names},
    }
    landmarks["left_foot"] = point
    pose = SimpleNamespace(valid=True, warnings=[], full_body_score=0.9, landmarks=landmarks)
    image = Image.fromarray(np.random.default_rng(1).integers(0, 256, (400, 320, 3), dtype=np.uint8))
    return QualityChecker(None).check_input(image, pose=pose)


@pytest.mark.parametrize("point", [(0.5, 1.1, 0.99), (-0.1, 0.9, 0.99), (1.0, 0.9, 0.99)])
def test_confident_offscreen_foot_is_rejected_despite_high_pose_score(point):
    result = check(point)
    assert not result["passed"]
    assert result["lower_body_framing"] == "cropped"
    assert any("다시 촬영" in issue for issue in result["issues"])


def test_full_body_inside_frame_passes():
    result = check()
    assert result["passed"]
    assert result["lower_body_framing"] == "visible"


@pytest.mark.parametrize("point", [(0.5, 1.1, 0.1), (float("nan"), 0.9, 0.99), None])
def test_uncertain_foot_is_not_mislabeled_as_crop(point):
    result = check(point)
    assert result["passed"]
    assert result["lower_body_framing"] == "uncertain"
