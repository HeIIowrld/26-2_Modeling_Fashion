"""Real-image, isolated GPU smoke test. Does not mark visual quality as passed."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import numpy as np
import torch
from PIL import Image
from catvton_tryon import CatVTONTryOn
from clothing_parser import ClothingParser
from config import garment_image_path
from pose_analyzer import PoseAnalyzer
from product_catalog import ProductCatalog
from schemas import Product, Recommendation
from shoe_tryon import OutfitTryOn, ShoeTryOn, foot_edit_mask


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('--people', type=Path, required=True)
    cli.add_argument('--catalog', type=Path, required=True)
    cli.add_argument('--model', type=Path, required=True)
    cli.add_argument('--output', type=Path, required=True)
    cli.add_argument('--shoe-manifest', type=Path)
    cli.add_argument('--prepare-only', action='store_true')
    cli.add_argument('--people-limit', type=int, default=2)
    cli.add_argument('--shoe-limit', type=int, default=2)
    cli.add_argument('--skip-outfit', action='store_true', help='의류+신발 전체 조합 1건을 생략한다')
    cli.add_argument('--prompt-file', type=Path, help='프롬프트 A/B용. 비교 실행에만 쓴다')
    cli.add_argument('--dump-parse', action='store_true', help='발 마스크 규칙을 GPU 없이 비교하려고 파싱 라벨을 저장한다')
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.prompt_file:
        import shoe_tryon
        shoe_tryon.PROMPT = args.prompt_file.read_text(encoding='utf-8').strip()
        (args.output / 'prompt.txt').write_text(shoe_tryon.PROMPT, encoding='utf-8')
    catalog = ProductCatalog(args.catalog).products
    shoes = [p for p in catalog if p.category == 'shoes' and garment_image_path(p.image_path).is_file()]
    if args.shoe_manifest:
        shoes = [Product(**p) for p in json.loads(args.shoe_manifest.read_text(encoding='utf-8'))]
    # Different products on the same person test actual reference conditioning.
    selected = []
    for item in shoes:
        if not selected or all(item.color != chosen.color for chosen in selected):
            selected.append(item)
        if len(selected) == args.shoe_limit:
            break
    if len(selected) < min(2, args.shoe_limit):
        raise RuntimeError('Two real shoe references with distinct catalog colors are required')
    parser = ClothingParser(use_fashn=True)
    pose_analyzer = PoseAnalyzer()
    eligible, rejected = [], []
    for path in sorted(args.people.iterdir()):
        if path.suffix.lower() not in {'.png', '.jpg', '.jpeg'}:
            continue
        try:
            pose = pose_analyzer.analyze(path)
            parsed = parser.parse(path, pose)
            mask = foot_edit_mask(parsed['segmentation'], pose)
            eligible.append((path, pose, parsed, mask))
            if args.dump_parse:
                Image.fromarray(np.asarray(parsed['segmentation']).astype(np.uint8)).save(
                    args.output / f'{path.stem}_parse.png')
                (args.output / f'{path.stem}_pose.json').write_text(
                    json.dumps({k: list(map(float, v)) for k, v in pose.landmarks.items()}), encoding='utf-8')
        except Exception as exc:
            rejected.append({'person': path.name, 'reason': str(exc)})
        if len(eligible) == args.people_limit:
            break
    pose_analyzer.close()
    (args.output / 'inputs.json').write_text(json.dumps({
        'eligible': [p.name for p, *_ in eligible], 'rejected': rejected,
        'shoes': [vars(p) for p in selected],
    }, ensure_ascii=False, indent=2))
    if not eligible:
        raise RuntimeError('No input passes both-foot visibility checks')
    for person, _, _, mask in eligible:
        Image.open(person).convert('RGB').save(args.output / f'{person.stem}_original.png')
        Image.fromarray(mask.astype(np.uint8) * 255).save(args.output / f'{person.stem}_mask.png')
    if args.prepare_only:
        from diffusers import Flux2KleinInpaintPipeline
        print({'pipeline': Flux2KleinInpaintPipeline.__name__, 'model_ready': ShoeTryOn(args.model).available,
               'eligible': [p.name for p, *_ in eligible]}, flush=True)
        return
    shoe_model = ShoeTryOn(args.model)
    clothing = CatVTONTryOn.fast(garment_cache_dir=args.output / 'garment_cache')
    clothing._garment_parser = parser
    adapter = OutfitTryOn(clothing, shoe_model, parser)
    records = []
    for person, pose, parsed, mask in eligible:
        context = {**parsed, 'pose': pose, 'parser': parser}
        for shoe in selected:
            reference = Path(shoe.image_path)
            if not reference.is_absolute():
                reference = garment_image_path(shoe.image_path)
            Image.open(reference).convert('RGB').save(args.output / f'{shoe.product_id}_reference.png')
            start = time.monotonic()
            output = args.output / f'{person.stem}_{shoe.product_id}.png'
            torch.cuda.reset_peak_memory_stats()
            try:
                adapter.generate(person, Recommendation(1, [shoe], 0, {}, []), output, context)
                record = {'person': person.name, 'products': [shoe.product_id], 'output': output.name,
                          'seconds': time.monotonic()-start, 'peak_vram_gb': torch.cuda.max_memory_allocated()/2**30,
                          'report': adapter.last_quality_reports, 'visual_review': 'pending'}
            except Exception as exc:
                record = {'person': person.name, 'products': [shoe.product_id], 'error': repr(exc)}
                import traceback
                traceback.print_exc()
            records.append(record)
            (args.output / 'results.json').write_text(json.dumps(records, ensure_ascii=False, indent=2))
            print(json.dumps(record, ensure_ascii=False), flush=True)
    # One complete outfit verifies that the two generators can share the GPU.
    if args.skip_outfit:
        (args.output / 'results.json').write_text(json.dumps(records, ensure_ascii=False, indent=2))
        if any('error' in r for r in records):
            raise RuntimeError('Some real GPU smoke cases failed; see results.json')
        return
    person, pose, parsed, _ = eligible[0]
    clothes = [next(p for p in catalog if p.category == c and garment_image_path(p.image_path).is_file())
               for c in ('top', 'bottom')]
    outfit = clothes + [selected[0]]
    start = time.monotonic()
    try:
        adapter.generate(person, Recommendation(1, outfit, 0, {}, []), args.output / 'complete_outfit.png',
                         {**parsed, 'pose': pose, 'parser': parser})
        record = {'person': person.name, 'products': [p.product_id for p in outfit],
                  'output': 'complete_outfit.png', 'seconds': time.monotonic()-start,
                  'report': adapter.last_quality_reports, 'visual_review': 'pending'}
    except Exception as exc:
        import traceback
        traceback.print_exc()
        record = {'person': person.name, 'products': [p.product_id for p in outfit], 'error': repr(exc)}
    records.append(record)
    (args.output / 'results.json').write_text(json.dumps(records, ensure_ascii=False, indent=2))
    if any('error' in r for r in records):
        raise RuntimeError('Some real GPU smoke cases failed; see results.json')


if __name__ == '__main__':
    main()
