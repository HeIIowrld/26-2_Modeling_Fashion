# 실행 환경과 검증 범위

2026-09-28 기준. CPU 테스트 통과와 GPU 이미지 생성 성공은 구분한다.

| 환경 | 버전 근거 | 검증 범위 |
| --- | --- | --- |
| Windows Python 3.11 로컬 | `requirements.txt`, `constraints-local-cpu.txt` | 설치된 torch 2.10.0+cpu, transformers 4.44.2, diffusers 0.30.3, accelerate 0.33.0에서 추천·API 회귀 검사. GPU 생성 미검증 |
| AMD ROCm Windows | `AMD_ROCM_SETUP.md`의 기존 별도 환경 | torch 2.9.1+rocm7.2.1, transformers 4.46.3, diffusers 0.31.0 설치 기록. 이번 작업에서 재검증하지 않음 |
| FLUX 신발 전용 | `requirements-shoe-runtime.txt`, `../../gpu_server/SHOES_VTON.md` | diffusers 0.40.0, transformers 5.15.1, accelerate 1.14.0. 서버 문서에 실제 생성 기록이 있음. 이번 작업은 미준비 오류 처리만 로컬 검증 |

새 CPU 환경에서는 Python 3.11 venv를 만들고 CPU torch/torchvision을 먼저 설치한다.
아래 명령은 저장소 루트에서 실행한다.

```powershell
python -m venv .venv-cpu
.venv-cpu/Scripts/python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu
.venv-cpu/Scripts/python -m pip install -r ai_fashion_recommender/requirements.txt -c ai_fashion_recommender/constraints-local-cpu.txt
.venv-cpu/Scripts/python -m pip check
```

GPU wheel은 플랫폼에 맞춰 설치한다. CPU 제약 파일로 GPU 서버를 덮어쓰지 않는다.
신발 전용 파일은 기본 CatVTON requirements와 함께 설치하지 않는다. 통합 GPU 서버의 환경을 재현하려면 기존 GPU 검증 환경의 전체 freeze와 체크포인트를 확보한 뒤 별도 환경에서 검증해야 한다.
공식 [FLUX.2 인페인팅 구현](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux2/pipeline_flux2_klein_inpaint.py)에도 해당 클래스가 있으며, 오래된 로컬 diffusers에 없다는 이유로 모든 환경에서 실행 불가라고 단정할 수 없다.

선택적 Notebook/Gradio 도구를 포함한 모든 전이 의존성을 잠근 파일은 아직 아니다. 새 venv 전체 설치와 GPU 생성은 이번 작업에서 실행하지 않았다.
기존 전역 환경의 `pip check`는 craftground↔protobuf 및 cvxpy↔numpy 충돌을 보고했다. 이번 작업은 패키지를 설치하거나 전역 환경을 변경하지 않았다. 시연 환경은 별도 venv로 유지한다.
