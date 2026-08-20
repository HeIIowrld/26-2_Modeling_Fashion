# AMD GPU (Windows)에서 돌리기

RX 7000/9000 시리즈는 AMD가 내놓은 Windows용 ROCm PyTorch 프리뷰로 가속할 수 있다.
**Python 3.12 전용**이라 기존 환경과 섞지 말고 별도 가상환경을 만든다.

```powershell
py -3.12 -m venv C:\venvs\fashion-gpu
C:\venvs\fashion-gpu\Scripts\python.exe -m pip install --no-cache-dir `
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_core-7.2.1-py3-none-win_amd64.whl `
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl `
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl `
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm-7.2.1.tar.gz
C:\venvs\fashion-gpu\Scripts\python.exe -m pip install --no-cache-dir `
    "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torch-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl" `
    "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torchvision-0.24.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl"
C:\venvs\fashion-gpu\Scripts\python.exe -m pip install "mediapipe==0.10.14" opencv-python `
    "transformers==4.46.3" "fashn-human-parser==0.1.1" "open-clip-torch==3.3.0" "ftfy==6.3.1" `
    "protobuf<5" "diffusers==0.31.0" accelerate gradio
```

ROCm 빌드도 `torch.cuda.is_available()`가 그대로 동작한다. 코드를 고치지 않고
NVIDIA 환경(Colab, HF Spaces)으로 그대로 옮길 수 있다.

## 걸렸던 문제

- **MIOpen BatchNorm 컴파일 실패** — ROCm Windows에서 SegFormer(FASHN 파서)가 GPU에서
  죽는다. `clothing_parser.py`가 ROCm + Windows를 감지하면 파서만 CPU로 돌린다.
  디퓨전(SD1.5)은 GroupNorm 기반이라 GPU에서 정상이다.
- **mediapipe 버전** — 1.0.0은 구형 `mp.solutions` API를 없애서 `pose_analyzer.py`가
  깨진다. 0.10.14로 고정한다.
- **transformers 5.x meta-init 충돌** — Marqo fashionSigLIP 원격 코드가 `__init__`에서
  가중치를 만들다 실패한다. `fashion_model.py`는 open_clip 직접 로딩을 쓴다.
- **torch 계열 추가 설치 주의** — `open_clip_torch`, `timm`을 그냥 설치하면 PyPI torch가
  ROCm torch를 덮어쓴다. `--no-deps`로 넣는다.
