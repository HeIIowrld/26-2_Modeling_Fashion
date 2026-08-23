"""계산 노드에 인터넷이 없으므로 master에서 미리 받아 공유 캐시에 넣는다."""
import sys, time
from huggingface_hub import snapshot_download

JOBS = [
    # (repo, allow_patterns, ignore_patterns, 설명)
    ("booksforcharlie/stable-diffusion-inpainting", None, None,
     "CatVTON 베이스 디퓨전"),
    ("zhengchong/CatVTON",
     ["mix-48k-1024/**", "*.json", "*.md", "*.txt"], None,
     "CatVTON attention (mix 버전만)"),
    ("fashn-ai/fashn-human-parser", None, None, "의류 분할 파서"),
    ("Marqo/marqo-fashionSigLIP", None, ["onnx/**"], "속성 백본 (ONNX 제외)"),
]

for repo, allow, ignore, desc in JOBS:
    print(f"\n=== {repo}  ({desc}) ===", flush=True)
    t = time.time()
    try:
        p = snapshot_download(repo, allow_patterns=allow, ignore_patterns=ignore,
                              max_workers=8)
        print(f"  완료 {time.time()-t:.0f}초 -> {p}", flush=True)
    except Exception as e:
        print(f"  실패 {type(e).__name__}: {str(e)[:200]}", flush=True)
        sys.exit(1)
print("\nALL_DONE", flush=True)
