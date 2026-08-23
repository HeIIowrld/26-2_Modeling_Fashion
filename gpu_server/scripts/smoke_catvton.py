"""팀원 datasets/ 없이 CatVTON 합성이 GPU에서 도는지 확인한다.
CatVTON 저장소의 데모 이미지를 쓰고, 마스크는 프로젝트의 PoseAnalyzer +
FASHN 파서로 만들어 실제 경로를 그대로 태운다."""
import time
from pathlib import Path
import numpy as np
from PIL import Image

PROJ = Path("/mnt/data1/dsl01/26-2_Modeling_Fashion")
DEMO = PROJ / "third_party/CatVTON/resource/demo/example/condition"
OUT = PROJ / "ai_fashion_recommender/outputs"
OUT.mkdir(parents=True, exist_ok=True)

person_p = sorted((DEMO / "person").glob("*.jpg"))[0]
garment_p = sorted((DEMO / "upper").glob("*.jpg"))[0]
print("인물 :", person_p.name)
print("의류 :", garment_p.name)

person = Image.open(person_p).convert("RGB").resize((768, 1024), Image.LANCZOS)
garment = Image.open(garment_p).convert("RGB")

print("\n[1/4] 포즈 분석", flush=True)
t = time.time()
from pose_analyzer import PoseAnalyzer
with PoseAnalyzer() as pa:
    pose = pa.analyze(person)
print(f"      {time.time()-t:.1f}초, 랜드마크 {len(getattr(pose, 'landmarks', []) or [])}개")

print("\n[2/4] FASHN 파서로 상의 마스크", flush=True)
t = time.time()
from clothing_parser import ClothingParser
res = ClothingParser(use_fashn=True).parse(person, pose)
print(f"      {time.time()-t:.1f}초, 검출 라벨: {res.get('present')}")
mask_arr = (np.asarray(res["upper_style_mask"]) > 0).astype(np.uint8) * 255
print(f"      마스크 비율 {mask_arr.mean()/255:.3f}")
if mask_arr.mean() == 0:
    raise SystemExit("마스크가 비었다")
mask = Image.fromarray(mask_arr).resize(person.size, Image.NEAREST)

print("\n[3/4] CatVTON 파이프라인 로드", flush=True)
t = time.time()
from catvton_tryon import CatVTONTryOn
tryon = CatVTONTryOn(num_inference_steps=20, max_retries=0)
tryon._load_pipeline()
print(f"      {time.time()-t:.1f}초, device={tryon.device}")

print("\n[4/4] 합성 실행 (20 steps)", flush=True)
t = time.time()
result = tryon._tryon_once(person, garment, mask)
dt = time.time() - t
dest = OUT / "smoke_catvton.jpg"
result.save(dest, quality=95)
person.save(OUT / "smoke_person.jpg", quality=95)
garment.save(OUT / "smoke_garment.jpg", quality=95)
print(f"      {dt:.1f}초 -> {dest}  크기 {result.size}")

import torch
print(f"\nVRAM 최대 사용: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
print("SMOKE_OK")
