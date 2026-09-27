#!/usr/bin/env python3
"""Research-only DiOr pilot: omit jacket layer, optionally replace the top.

Uses DeepFashion-MM's official person parsing and an independently inferred
MediaPipe/OpenPose-18 heatmap. DiOr is a low-resolution, external baseline, not
a production dependency. DeepFashion-MM examples must not be redistributed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image


SIZE = (176, 256)  # DiOr's (width, height), not the source photo's size.
PERSON_IDS = [0, 4, 6, 7]  # background, face, arm, leg
WITH_JACKET = [2, 5, 1, 3]  # hair, top, bottom, jacket
WITHOUT_JACKET = [2, 5, 1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def map_official_parsing(official: np.ndarray) -> np.ndarray:
    """Map the relevant DeepFashion-MM classes to DiOr's eight slots.

    This is an experimental approximation. DiOr's training LIP map assigns
    Coat to slot 3, although that slot is named Skirt in its dataset code.
    """
    if official.ndim != 2:
        raise ValueError("expected one-channel official segmentation")
    parsed = np.zeros(official.shape, dtype=np.uint8)
    parsed[np.isin(official, (1, 3))] = 5  # top/dress
    parsed[official == 2] = 3  # outerwear -> DiOr jacket slot
    parsed[np.isin(official, (4, 5, 6))] = 1  # skirt, pants, leggings
    parsed[np.isin(official, (7, 13))] = 2  # headwear/hair
    parsed[official == 14] = 4  # face
    parsed[official == 15] = 6  # visible skin, mostly arms/hands
    parsed[official == 11] = 7  # footwear
    return parsed


def map_lip_parsing(lip: np.ndarray) -> np.ndarray:
    """Use DiOr's published LIP-to-eight-class training mapping exactly.

    Upstream maps LIP arms to its ``Leg`` slot and LIP legs to ``Arm``;
    preserving that counterintuitive map keeps this pilot checkpoint-aligned.
    """
    if lip.ndim != 2:
        raise ValueError("expected one-channel LIP segmentation")
    parsed = np.zeros(lip.shape, dtype=np.uint8)
    parsed[np.isin(lip, (5, 6))] = 5  # upper-clothes/dress
    parsed[lip == 7] = 3  # coat -> upstream's jacket slot
    parsed[np.isin(lip, (9, 12))] = 1  # pants/skirt
    parsed[lip == 2] = 2  # hair
    parsed[lip == 13] = 4  # face
    parsed[np.isin(lip, (16, 17))] = 6  # upstream: legs -> Arm
    parsed[np.isin(lip, (14, 15))] = 7  # upstream: arms -> Leg
    return parsed


def pose_heatmaps(rgb: np.ndarray, *, visibility: float = 0.35) -> tuple[np.ndarray, int]:
    """Convert MediaPipe landmarks to DiOr/OpenPose-18 Gaussian heatmaps."""
    landmark = mp.solutions.pose.PoseLandmark
    with mp.solutions.pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.45,
    ) as detector:
        result = detector.process(rgb)
    if not result.pose_landmarks:
        raise ValueError("MediaPipe could not find a person pose")
    raw = result.pose_landmarks.landmark
    pairs = [
        landmark.NOSE,
        None,  # neck = midpoint of both shoulders
        landmark.RIGHT_SHOULDER, landmark.RIGHT_ELBOW, landmark.RIGHT_WRIST,
        landmark.LEFT_SHOULDER, landmark.LEFT_ELBOW, landmark.LEFT_WRIST,
        landmark.RIGHT_HIP, landmark.RIGHT_KNEE, landmark.RIGHT_ANKLE,
        landmark.LEFT_HIP, landmark.LEFT_KNEE, landmark.LEFT_ANKLE,
        landmark.LEFT_EYE, landmark.RIGHT_EYE,
        landmark.LEFT_EAR, landmark.RIGHT_EAR,
    ]
    height, width = SIZE[1], SIZE[0]
    yy, xx = np.mgrid[:height, :width]
    heatmaps = np.zeros((18, height, width), dtype=np.float32)
    visible = 0
    for index, key in enumerate(pairs):
        if key is None:
            a, b = raw[landmark.RIGHT_SHOULDER], raw[landmark.LEFT_SHOULDER]
            if min(a.visibility, b.visibility) < visibility:
                continue
            x, y = (a.x + b.x) / 2, (a.y + b.y) / 2
        else:
            point = raw[key]
            if point.visibility < visibility:
                continue
            x, y = point.x, point.y
        if not 0 <= x < 1 or not 0 <= y < 1:
            continue
        visible += 1
        heatmaps[index] = np.exp(-((xx - x * width) ** 2 + (yy - y * height) ** 2) / 72.0)
    if visible < 8:
        raise ValueError(f"only {visible}/18 usable pose landmarks")
    return heatmaps, visible


def load_person(
    image_path: Path, parsing_path: Path, *, taxonomy: str = "official"
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    with Image.open(image_path) as image:
        original = np.asarray(image.convert("RGB"))
        rgb = np.asarray(image.convert("RGB").resize(SIZE, Image.Resampling.BILINEAR))
    with Image.open(parsing_path) as parsing:
        official = np.asarray(parsing.resize(SIZE, Image.Resampling.NEAREST), dtype=np.uint8)
    if official.ndim != 2:
        raise ValueError(f"not a one-channel parsing PNG: {parsing_path}")
    pose, visible = pose_heatmaps(original)
    if taxonomy == "official":
        mapped = map_official_parsing(official)
    elif taxonomy == "lip":
        mapped = map_lip_parsing(official)
    else:
        raise ValueError(f"unknown parsing taxonomy: {taxonomy}")
    return rgb, mapped, pose, visible


def model_options(checkpoint_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        isTrain=False, phase="test", n_human_parts=8, n_kpts=18,
        style_nc=64, n_style_blocks=4, netG="dior", netE="adgan", ngf=64,
        norm_type="instance", relu_type="leakyrelu", init_type="orthogonal",
        init_gain=0.02, gpu_ids=[0], frozen_flownet=True, random_rate=1,
        perturb=False, warmup=False, name="DIOR_64", vgg_path="",
        flownet_path="", checkpoints_dir=str(checkpoint_dir), frozen_enc=True,
        load_iter=0, epoch="latest", verbose=False,
    )


def to_tensor(rgb: np.ndarray, torch):
    array = rgb.astype(np.float32).transpose(2, 0, 1) / 127.5 - 1.0
    return torch.from_numpy(array[None]).cuda()


def save_tensor(tensor, path: Path) -> None:
    array = tensor[0].detach().float().cpu().numpy().transpose(1, 2, 0)
    Image.fromarray(np.uint8(np.clip((array + 1) * 127.5, 0, 255))).save(path)


def install_inference_only_transform(torch) -> None:
    """Provide DiOr's flow warp without importing its training dataset stack.

    The upstream ``utils.train_utils`` module imports pandas and scikit-image
    only for training helpers. Its inference path uses this one function.
    """
    module = ModuleType("utils.train_utils")

    def torch_transform(img, flow_field):
        batch, _, height, width = img.size()
        flow_field = torch.nn.functional.interpolate(flow_field, (height, width), mode="bilinear")
        x = torch.arange(width, device=img.device).view(1, -1).expand(height, -1).float()
        y = torch.arange(height, device=img.device).view(-1, 1).expand(-1, width).float()
        grid = torch.stack((2 * x / (width - 1) - 1, 2 * y / (height - 1) - 1), dim=0)
        grid = grid.unsqueeze(0).expand(batch, -1, -1, -1)
        flow_x = (2 * flow_field[:, 0] / (width - 1)).view(batch, 1, height, width)
        flow_y = (2 * flow_field[:, 1] / (height - 1)).view(batch, 1, height, width)
        flow = torch.cat((flow_x, flow_y), dim=1)
        return torch.nn.functional.grid_sample(img, (grid + flow).permute(0, 2, 3, 1))

    module.torch_transform = torch_transform
    sys.modules[module.__name__] = module


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--dior-repo", type=Path, required=True)
    cli.add_argument("--checkpoint-dir", type=Path, required=True)
    cli.add_argument("--person", type=Path, required=True)
    cli.add_argument("--person-segmentation", type=Path, required=True)
    cli.add_argument("--person-parsing-taxonomy", choices=("official", "lip"), default="official")
    cli.add_argument("--top-source", type=Path, required=True)
    cli.add_argument("--top-segmentation", type=Path, required=True)
    cli.add_argument("--out", type=Path, required=True)
    args = cli.parse_args()
    for path in (args.person, args.person_segmentation, args.top_source, args.top_segmentation):
        if not path.is_file():
            cli.error(f"missing input: {path}")
    if not (args.checkpoint_dir / "DIOR_64" / "latest_net_G.pth").is_file():
        cli.error("missing DiOr checkpoint")
    if not (args.dior_repo / "models" / "dior_model.py").is_file():
        cli.error("missing DiOr repository")
    args.out.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    sys.path.insert(0, str(args.dior_repo))
    import torch  # noqa: E402
    install_inference_only_transform(torch)
    from models.dior_model import DIORModel  # noqa: E402

    if not torch.cuda.is_available():
        raise RuntimeError("DiOr pilot requires a CUDA GPU")
    # DiOr's load_networks() uses torch.load without a safety setting.
    original_load = torch.load

    def safe_load(*args, **kwargs):
        kwargs.setdefault("weights_only", True)
        return original_load(*args, **kwargs)

    torch.load = safe_load
    model = DIORModel(model_options(args.checkpoint_dir))
    model.load_networks("latest")
    for name in model.model_names:
        getattr(model, "net" + name).eval()

    person_rgb, person_parse, person_pose, person_visible = load_person(
        args.person, args.person_segmentation, taxonomy=args.person_parsing_taxonomy
    )
    source_rgb, source_parse, source_pose, source_visible = load_person(
        args.top_source, args.top_segmentation
    )
    if int((person_parse == 3).sum()) < 100:
        raise ValueError("person parsing has no visible outerwear label")
    if int((source_parse == 5).sum()) < 100:
        raise ValueError("top source parsing has no visible top label")
    Image.fromarray(person_rgb).save(args.out / "person.png")
    Image.fromarray(source_rgb).save(args.out / "top_source.png")
    Image.fromarray(person_parse).save(args.out / "person_parse_dior.png")
    with torch.inference_mode():
        pimg = to_tensor(person_rgb, torch)
        pparse = torch.from_numpy(person_parse[None]).cuda()
        ppose = torch.from_numpy(person_pose[None]).cuda()
        simg = to_tensor(source_rgb, torch)
        sparse = torch.from_numpy(source_parse[None]).cuda()
        spose = torch.from_numpy(source_pose[None]).cuda()
        person_segments = model.encode_attr(pimg, pparse, ppose, ppose, PERSON_IDS)
        garment_segments = model.encode_attr(pimg, pparse, ppose, ppose)
        source_top = model.encode_single_attr(simg, sparse, spose, ppose, i=5)
        conditions = {
            "with_jacket_original_top": [garment_segments[i] for i in WITH_JACKET],
            "without_jacket_original_top": [garment_segments[i] for i in WITHOUT_JACKET],
            "without_jacket_target_top": [
                source_top if i == 5 else garment_segments[i] for i in WITHOUT_JACKET
            ],
        }
        for name, layers in conditions.items():
            generated = model.netG(ppose, person_segments, layers)
            save_tensor(generated, args.out / f"{name}.png")
    record = {
        "model": "Dressing in Order / DIOR_64",
        "dior_commit": "ee9678dbc724bfd1f548fb89629c6fe33f1ef572",
        "resolution": [SIZE[0], SIZE[1]],
        "person": str(args.person),
        "person_sha256": sha256(args.person),
        "person_parsing_taxonomy": args.person_parsing_taxonomy,
        "person_segmentation_sha256": sha256(args.person_segmentation),
        "top_source": str(args.top_source),
        "top_source_sha256": sha256(args.top_source),
        "top_segmentation_sha256": sha256(args.top_segmentation),
        "checkpoint_g_sha256": sha256(args.checkpoint_dir / "DIOR_64" / "latest_net_G.pth"),
        "person_outer_fraction": round(float(np.mean(person_parse == 3)), 6),
        "person_visible_top_fraction": round(float(np.mean(person_parse == 5)), 6),
        "source_visible_top_fraction": round(float(np.mean(source_parse == 5)), 6),
        "person_pose_points": person_visible,
        "source_pose_points": source_visible,
        "seconds": round(time.perf_counter() - start, 2),
        "outputs": {name: sha256(args.out / f"{name}.png") for name in conditions},
    }
    (args.out / "result.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
