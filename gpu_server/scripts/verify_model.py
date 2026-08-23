"""채택 모델(augmented)이 실제 로더로 열리는지 확인한다."""
import hashlib, json
from pathlib import Path
from config import FASHION_ATTRIBUTE_HEADS_PATH as P

print("로드 대상 :", P)
print("sha256    :", hashlib.sha256(Path(P).read_bytes()).hexdigest())
m = json.load(open(Path(P).with_suffix(".metrics.json"), encoding="utf-8"))
print("checkpoint:", m.get("checkpoint"))
print("학습 crop :", m.get("training", {}).get("train_crops_total"))

import torch
from fashion_attribute_model import load_attribute_heads
dev = "cuda" if torch.cuda.is_available() else "cpu"
ck = load_attribute_heads(P, device=dev)

def describe(obj, depth=0):
    if isinstance(obj, dict):
        return {k: type(v).__name__ for k, v in obj.items()}
    return type(obj).__name__

print("로더 반환 :", describe(ck))
if isinstance(ck, dict):
    tasks = ck.get("tasks") or ck.get("task_names") or ck.get("schema")
    if tasks is not None:
        try: print("태스크 수 :", len(tasks))
        except TypeError: pass
    pre = ck.get("preprocessing") or ck.get("task_preprocessing")
    if pre: print("전처리 라우팅:", dict(list(pre.items())[:6]), "...")
print("device    :", dev)
print("MODEL_OK")
