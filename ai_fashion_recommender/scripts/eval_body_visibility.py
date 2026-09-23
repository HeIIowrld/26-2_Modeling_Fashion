"""Run the body-visibility gate on real photos and record every signal it used.

Measures the gate's behaviour on photos, not code paths: how many photos whose
clothes do show the body line get blocked. Labels come from the manifest and are
written by a human or by the photo's own dataset annotation; this script never
manufactures them. It also records the three silhouette widths and the resulting
body-shape label so the same run can answer "would this distortion flip the
class?" without re-running the models.

Manifest: [{"id": "...", "image": "<relative to manifest>", "group": "pants",
            "truth": "fitted|loose|dress", "labels": {...}}, ...]
`group`, `truth` and `labels` are optional and are copied to the output as-is.

usage: eval_body_visibility.py --repo REPO --manifest M --gate PATH --output OUT
  --gate: the body_visibility.py under test (a branch copy, not necessarily --repo's)
"""
import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

LABEL_FIELDS = ("group", "truth", "labels")


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('--repo', type=Path, required=True)
    cli.add_argument('--manifest', type=Path, required=True)
    cli.add_argument('--gate', type=Path, required=True)
    cli.add_argument('--output', type=Path, required=True)
    cli.add_argument('--segmentation', action='store_true',
                     help='also save each segmentation map (large; for visual review sheets)')
    args = cli.parse_args()
    sys.path.insert(0, str(args.repo / 'ai_fashion_recommender/src'))
    import numpy as np
    from body_measure import BodyMeasurementEstimator
    from body_shape import classify_from_circumferences
    from clothing_parser import ClothingParser
    from config import FASHION_ATTRIBUTE_HEADS_PATH
    from fashion_model import FashionClassifier
    from outfit_analyzer import OutfitAnalyzer
    from pose_analyzer import PoseAnalyzer
    from quality_checker import QualityChecker
    spec = importlib.util.spec_from_file_location('evaluated_gate', args.gate)
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    pose_model = PoseAnalyzer(model_complexity=1)
    parser = ClothingParser(use_fashn=True)
    classifier = FashionClassifier(enabled=True, device='cpu', attribute_checkpoint=FASHION_ATTRIBUTE_HEADS_PATH)
    analyzer = OutfitAnalyzer(parser, classifier)
    checker = QualityChecker(pose_model)
    estimator = BodyMeasurementEstimator()
    rows = json.loads(args.manifest.read_text(encoding='utf-8'))
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {'device': classifier.device, 'parser': parser.backend,
                'checkpoint': str(FASHION_ATTRIBUTE_HEADS_PATH),
                'checkpoint_sha256': hashlib.sha256(FASHION_ATTRIBUTE_HEADS_PATH.read_bytes()).hexdigest(),
                'gate_sha256': hashlib.sha256(args.gate.read_bytes()).hexdigest(),
                'manifest_sha256': hashlib.sha256(args.manifest.read_bytes()).hexdigest()}
    (args.output/'runtime.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    # Resume instead of restarting: a few hundred photos take longer than one
    # interactive session on a shared node.
    predictions = args.output/'predictions.jsonl'
    done = ({json.loads(line)['id'] for line in predictions.read_text(encoding='utf-8').splitlines()}
            if predictions.exists() else set())
    try:
        for row in rows:
            if row['id'] in done:
                continue
            started = time.monotonic()
            result = {'id': row['id'], **{key: row[key] for key in LABEL_FIELDS if key in row}}
            try:
                path = args.manifest.parent / row['image']
                pose = pose_model.analyze(path)
                quality = checker.check_input(path, pose=pose)
                result['basic_quality'] = quality
                # Also assess pose-valid photos rejected by framing/blur, to
                # separate clothing-gate behaviour from earlier input gates.
                if pose.valid:
                    outfit, parsed = analyzer.analyze(path, pose)
                    visibility = gate.assess_body_visibility(outfit, parsed)
                    seg = np.asarray(parsed['segmentation'])
                    person = int(np.count_nonzero(seg)) or 1
                    measured = estimator.estimate(path, pose, height_cm=None)
                    shape, _ = classify_from_circumferences(
                        measured.chest_width, measured.waist_width, measured.hip_width)
                    sources = getattr(outfit, 'attribute_sources', {})
                    result.update({
                        'visibility': visibility, 'status': visibility['status'],
                        'outfit': outfit.to_dict(), 'segmentation_backend': parsed['backend'],
                        # Flattened so the report can aggregate without the model stack.
                        'fit': getattr(outfit, 'fit', ''), 'lower_fit': getattr(outfit, 'lower_fit', ''),
                        'fit_source': sources.get('fit', 'mask'),
                        'lower_fit_source': sources.get('lower_fit', 'mask'),
                        'upper_type': getattr(outfit, 'upper_type', ''),
                        'outer_category': getattr(outfit, 'outer_category', ''),
                        'skirt_fraction': round(float(np.count_nonzero(np.isin(seg, (4, 5))) / person), 4),
                        'chest': round(measured.chest_width, 2), 'waist': round(measured.waist_width, 2),
                        'hip': round(measured.hip_width, 2), 'shape': shape,
                        'waist_chest': round(measured.waist_width / max(measured.chest_width, 1e-6), 4),
                        'hip_chest': round(measured.hip_width / max(measured.chest_width, 1e-6), 4),
                    })
                    if args.segmentation:
                        np.savez_compressed(args.output/f"{row['id']}.npz", segmentation=seg)
                else:
                    result['visibility'] = None
                    result['error'] = 'pose'
                result['end_to_end_passed'] = bool(
                    quality['passed'] and result['visibility'] and result['visibility']['passed'])
            except Exception as exc:  # noqa: BLE001  하나가 실패해도 남은 표본을 계속 측정한다
                result['error'] = f'{type(exc).__name__}: {exc}'
            result['seconds'] = round(time.monotonic()-started, 2)
            with predictions.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(result, ensure_ascii=False)+'\n')
            print(row['id'], result['seconds'], result.get('status', result.get('error')), flush=True)
    finally:
        pose_model.close()


if __name__ == '__main__':
    main()
