# 체형·목적 기반 AI 코디 추천 1단계

`main.ipynb`가 전체 실행 순서를 담당하고 각 기능은 같은 폴더의 Python 모듈로 분리되어 있습니다.

## 구조

- `main.ipynb`: 사용자 입력부터 추천 결과까지 단계별 실행
- `pose_analyzer.py`: MediaPipe Pose 기반 체형·자세 참고 비율
- `clothing_parser.py`: FASHN Human Parser로 의류 종류별 픽셀 마스크 생성
- `garment_attribute_analyzer.py`: 의류 마스크와 관절 위치를 결합해 소매·상의·하의 길이 및 핏 추정
- `fashion_model.py`: FashionSigLIP으로 스타일·패턴·소재 후보 분류
- `fashion_prompts.py`: DeepFashion-MultiModal 라벨에 맞춘 패턴·소재·네크라인 후보
- `outfit_analyzer.py`: 의류 색상과 세부 속성 분석 결과 통합
- `deepfashion_dataset.py`: 공식 MultiModal 텍스트 라벨 로딩과 현재 모델 정확도 평가
- `deepfashion_evaluation.ipynb`: DeepFashion 데이터 경로 설정부터 평가 보고서 저장까지 단계별 실행
- `recommendation_engine.py`: 규칙 기반 후보 점수 계산과 설명 생성
- `product_catalog.py`: 로컬 샘플 상품 카탈로그
- `virtual_tryon.py`: VTON 교체용 인터페이스와 비합성 추천 보드
- `quality_checker.py`: 입력·합성 결과 품질 검사
- `feedback_store.py`: 사용자 피드백 JSONL 저장
- `musinsa_crawler.py`: 무신사 상품 메타데이터·전면샷 수집기 (연구용 소규모)
- `app.py`: 사진 업로드 → 분석 → 무신사 추천을 제공하는 Gradio 웹 앱

## 웹 앱 실행

```bash
python musinsa_crawler.py --per-category 60   # 카탈로그 수집 (data/products_musinsa.csv)
python app.py                                 # 웹 앱 실행 (http://localhost:7860)
python app.py --light                         # 무거운 모델 없이 UI 흐름만 확인
python app.py --share                         # 팀원용 임시 공개 링크
```

- `data/products_musinsa.csv`가 있으면 자동으로 무신사 카탈로그를 사용하고, 없으면 `data/products.csv` 샘플을 사용합니다.
- 크롤링은 연구·학습용 소규모로만 사용하고 수집한 이미지는 재배포하지 않습니다.

## 실제 가상 피팅 (CatVTON)

`python app.py --vton`으로 실행하면 추천 보드 대신 CatVTON 디퓨전 모델이 추천 옷을 실제로 입힌 합성 사진을 생성합니다.

- 준비물: 프로젝트 상위 폴더에 `third_party/CatVTON` 클론
  (`git clone https://github.com/Zheng-Chong/CatVTON.git`)
- 마스크는 CatVTON의 AutoMasker(detectron2 필요) 대신 프로젝트의 FASHN 마스크를 사용하므로 Windows에서도 동작합니다.
- 첫 실행 시 HuggingFace에서 체크포인트 약 5GB를 내려받습니다.
- CPU에서는 장당 수 분 이상 걸리므로 GPU 환경을 권장합니다.
- CatVTON 가중치는 CC BY-NC-SA 4.0(비상업) 라이선스입니다.

### AMD GPU (Windows) 환경

RX 7000/9000 시리즈는 AMD의 Windows용 ROCm PyTorch 프리뷰로 GPU 가속이 가능합니다. **Python 3.12 전용**이므로 별도 가상환경을 만듭니다.

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

ROCm 빌드는 `torch.cuda.is_available()`가 그대로 동작하므로 코드 수정 없이 NVIDIA 환경(Colab, HF Spaces)으로 이식됩니다.

## 실행

1. 팀원은 `ai_fashion_recommender` 폴더 전체를 자신의 PC로 복사합니다.
2. `pip install -r requirements.txt`로 필요한 패키지를 설치합니다.
3. `main.ipynb`를 열고 맨 위의 **팀원별 로컬 경로 설정** 셀을 수정합니다.
4. 사용 조건을 입력한 뒤 Notebook을 위에서 아래로 실행합니다.

### 로컬 경로 설정

Notebook에서는 다음 다섯 값만 수정하면 됩니다.

```python
PROJECT_DIR_INPUT = r''
IMAGE_PATH_INPUT = r'data/input_person.jpg'
DATA_DIR_INPUT = r''
OUTPUT_DIR_INPUT = r''
FONT_PATH_INPUT = r''
```

- 빈 `PROJECT_DIR_INPUT`은 현재 작업 폴더와 그 아래 `ai_fashion_recommender`를 자동 탐색합니다.
- 상대경로는 프로젝트 폴더를 기준으로 처리하므로 팀 공유에는 `data/input_person.jpg` 같은 형식을 권장합니다.
- 프로젝트를 자동으로 찾지 못할 때만 `PROJECT_DIR_INPUT`에 각자 프로젝트 절대경로를 입력합니다.
- `DATA_DIR_INPUT`과 `OUTPUT_DIR_INPUT`은 비워두면 각각 프로젝트 내부 `data`, `outputs`를 사용합니다.
- 한글이 깨지는 환경에서는 `FONT_PATH_INPUT`에 설치된 한글 `.ttf` 또는 `.ttc` 파일을 지정합니다.

Notebook 밖에서 모듈만 사용할 때는 `FASHION_DATA_DIR`, `FASHION_OUTPUT_DIR`, `FASHION_FONT_PATH` 환경변수로 같은 경로를 변경할 수 있습니다. 별도 설정이 없으면 모든 경로는 `config.py`가 있는 프로젝트 폴더를 기준으로 결정됩니다.

기본 설정은 FASHN Human Parser와 FashionSigLIP을 모두 사용합니다. 첫 실행에서는 체크포인트를 내려받기 때문에 시간이 오래 걸릴 수 있고, 이후에는 로컬 캐시를 사용합니다. CPU에서도 실행할 수 있지만 FashionSigLIP 분석은 수십 초가 걸릴 수 있습니다.

FASHN 결과가 있어야 상·하의 영역과 옷 길이를 분석합니다. 포즈 기반 대체 마스크는 모델 연결 문제를 확인하기 위한 디버깅 수단이며 정식 결과로 취급하지 않습니다.

## DeepFashion으로 분석기 평가하기

DeepFashion 이미지는 프로젝트에 포함하지 않습니다. 공식 사용 동의 절차를 거쳐 `DeepFashion-MultiModal`을 받은 뒤 `deepfashion_evaluation.ipynb`의 경로 셀에서 다음 파일 위치를 지정합니다.

- 이미지 폴더
- shape 라벨 파일
- fabric 라벨 파일
- color/pattern 라벨 파일

평가 Notebook은 현재 모델의 소매 길이, 하의 길이, 패턴, 소재, 네크라인 정확도와 대표 오답을 `outputs/deepfashion_evaluation.json`에 저장합니다. CPU에서는 `MAX_SAMPLES=20`처럼 작은 수로 먼저 확인하는 것을 권장합니다.

## 현재 한계

- 사진 기반 체형 비율은 실제 신체 치수가 아닙니다.
- 샘플 패션 규칙은 전문가 검증 전의 MVP 규칙입니다.
- 상품 CSV는 추천 로직 검증용이며 실제 쇼핑몰 상품이 아닙니다.
- 현재 출력은 실제 가상 피팅이 아니라 명시적으로 표시된 추천 보드입니다.
- IDM-VTON은 비상업 라이선스이므로 상용화 전 별도 모델·라이선스 검토가 필요합니다.
- FASHN Human Parser는 NVIDIA SegFormer 라이선스, FashionSigLIP 체크포인트는 Apache-2.0 조건을 각각 확인해야 합니다.
- DeepFashion 및 DeepFashion-MultiModal은 비상업 연구 전용이며 원본과 파생 데이터의 재배포가 제한됩니다.
