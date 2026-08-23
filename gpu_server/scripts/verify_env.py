import importlib, sys
import torch
print("torch      :", torch.__version__)
print("CUDA 사용  :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU        :", torch.cuda.get_device_name(0))
    print("VRAM       : %.1f GB" % (torch.cuda.get_device_properties(0).total_memory / 1e9))
    x = torch.randn(2000, 2000, device="cuda")
    print("행렬곱 검증 :", bool(float((x @ x).sum()) != 0.0))

print("\n--- 패키지 ---")
for m in ("numpy", "cv2", "mediapipe", "transformers", "diffusers",
          "open_clip", "fashn_human_parser", "PIL"):
    try:
        mod = importlib.import_module(m)
        print("  %-20s OK   %s" % (m, getattr(mod, "__version__", "")))
    except Exception as e:
        print("  %-20s 실패 %s: %s" % (m, type(e).__name__, str(e)[:100]))

print("\n--- 프로젝트 모듈 ---")
for m in ("config", "clothing_parser", "fashion_model", "outfit_analyzer",
          "recommendation_engine", "catvton_tryon"):
    try:
        importlib.import_module(m)
        print("  %-22s OK" % m)
    except Exception as e:
        print("  %-22s 실패 %s: %s" % (m, type(e).__name__, str(e)[:110]))
