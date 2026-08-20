# 체형·목적 기반 AI 코디 추천 1단계

`main.ipynb`가 전체 실행 순서를 담당하고 각 기능은 같은 폴더의 Python 모듈로 분리되어 있습니다.

## 구조

- `main.ipynb`: 사용자 입력부터 추천 결과까지 단계별 실행
- `pose_analyzer.py`: MediaPipe Pose 기반 체형·자세 참고 비율
- `clothing_parser.py`: FASHN Human Parser로 의류 종류별 픽셀 마스크 생성
- `garment_attribute_analyzer.py`: 의류 마스크와 관절 위치를 결합해 소매·상의·하의 길이 및 핏 추정
- `fashion_model.py`: FashionSigLIP과 학습된 다중 속성 헤드 추론
- `fashion_attribute_schema.py`: 속성별 라벨과 단일·복수 분류 방식 정의
- `fashion_attribute_dataset.py`: 학습 CSV 로딩과 Fashionpedia 주석 변환
- `prepare_fashionpedia_seed.py`: Fashionpedia 공식 세부 속성 주석과 이미지 추출
- `prepare_fashion200k_supplement.py`: Fashion200K 상품 메타데이터에서 셔츠·블라우스·폴로·소재 보완 샘플 구성
- `prepare_fashion200k_bottoms.py`: Fashion200K 상품명에서 하의 종류·다리 모양·기장·구조 디테일 보완 샘플 구성
- `fashion_attribute_model.py`: 고정 FashionSigLIP 특징 위의 속성별 분류 헤드
- `fashion_attribute_training.py`: 특징 캐시, 헤드 학습, 평가, 임계값 선택
- `train_fashion_attribute_heads.ipynb`: 데이터 준비부터 체크포인트 생성까지 단계별 학습
- `fashion_prompts.py`: DeepFashion-MultiModal 라벨에 맞춘 패턴·소재·네크라인 후보
- `outfit_analyzer.py`: 의류 색상과 세부 속성 분석 결과 통합
- `deepfashion_dataset.py`: 공식 MultiModal 텍스트 라벨 로딩과 현재 모델 정확도 평가
- `deepfashion_evaluation.ipynb`: DeepFashion 데이터 경로 설정부터 평가 보고서 저장까지 단계별 실행
- `fashion_rules.py`: `FASHION_RULES_MASTER.md`의 R-* 규칙 ID와 메타데이터 로딩
- `recommendation_engine.py`: 활성 패션 규칙 기반 후보 필터링·점수 계산·설명 생성
- `product_catalog.py`: 로컬 샘플 상품 카탈로그
- `virtual_tryon.py`: VTON 교체용 인터페이스와 비합성 추천 보드
- `catvton_tryon.py`: CatVTON 디퓨전 기반 실제 가상 피팅 어댑터 (FASHN 마스크 사용)
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
- `models/fashion_attribute_heads.pt`가 있으면 학습된 속성 헤드로 옷 종류·속성을 인식하고, 없으면 zero-shot 분류로 자동 대체합니다.

## 실제 가상 피팅 (CatVTON)

`python app.py --vton`으로 실행하거나 `main.ipynb`에서 `USE_VTON = True`로 바꾸면 추천 보드 대신 CatVTON 디퓨전 모델이 추천 옷을 실제로 입힌 합성 사진을 생성합니다.

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

### 합성 품질: 확인된 실패 모드와 자동 점검

같은 사람·같은 마스크·같은 시드에서 상품만 바꾼 통제 실험(2026-08-20, `female_012`)으로 두 가지 실패 원인을 분리해 확인했습니다. 둘 다 디퓨전 모델의 성능 문제가 아니라 **입력 조건** 문제입니다.

**1. 레퍼런스 해상도 부족 → 시스루·뭉개짐**

상품 대표 이미지가 전신 착용컷이면 거기서 잘라낸 옷 조각이 작아서, 조건 입력 해상도로 확대하는 과정에 텍스처가 뭉개지고 결과가 반투명하게 렌더링됩니다.

| 레퍼런스 | coverage | 결과 |
|---|---|---|
| MS6797005 (착용 전신컷에서 추출한 셔츠) | 0.15 | 시스루 |
| MS6677115 (상품 클로즈업) | 충분 | 불투명하게 정상 합성 |

`coverage`는 *조건 입력 픽셀 수 대비 원본에 실제로 존재하는 의류 픽셀 수*입니다(`evaluate_garment_reference`). 1.0 미만이면 확대가 일어납니다. `min_reference_coverage`(기본 0.25) 밑이면 경고를 남깁니다.

**2. 새 옷이 원래 옷보다 짧을 때 → 마스크가 옷 텍스처로 채워짐**

마스크는 *원래 입고 있던 옷* 기준으로 만들어집니다. 더 짧은 옷을 넣으면 남는 마스크 영역(종아리·팔)을 모델이 맨살이 아니라 옷 비슷한 것으로 채웁니다.

| 기장 차이 | 결과 |
|---|---|
| 긴바지 → 미디·7부 (gap=1) | 정상 |
| 긴바지 → 쇼츠·미니 (gap=3) | 다리 전체가 니트 텍스처로 덮임 |

소매도 같은 원리입니다(긴팔 마스크에 반팔 상품을 넣으면 긴팔로 그려짐). `bottom_length_gap` / `sleeve_length_gap`이 `UNRELIABLE_LENGTH_GAP`(2단계) 이상이면 경고합니다.

**점검 결과 사용**

`generate()`에 `context={"outfit": ..., "classifier": ...}`를 함께 넘기면 위 두 점검이 돌고 사유가 `tryon.last_warnings`에 쌓입니다. 웹 앱은 분석 패널에, Notebook은 결과 이미지 아래에 표시합니다. **경고가 있는 결과는 합성이 깨졌을 수 있으므로 그대로 신뢰하면 안 됩니다.**

근본 해결은 카탈로그에 **상품 단독컷을 확보**하는 것입니다. 착용컷만 있는 상품은 VTON 렌더링 대상에서 제외하거나 단독컷을 다시 수집해야 합니다.

**아직 해결되지 않은 것**

- FASHN이 옷 위의 가방끈을 `top`으로 분류하는 경우가 있어(MS6797005: 어두운 픽셀 6.5k개) 세그멘테이션 기반 제거가 안 됩니다. 가방·머리카락·팔로 분류되는 가림 요소만 정제 단계에서 제외합니다.
- 마스크 모양 프라이어 자체는 CatVTON 구조상 우회할 수 없습니다. 위 2번은 감지·안내만 하고 합성 자체를 고치지는 못합니다.

## 실행

1. 팀원은 `ai_fashion_recommender` 폴더 전체를 자신의 PC로 복사합니다.
2. `pip install -r requirements.txt`로 필요한 패키지를 설치합니다.
3. 여러 Python이 설치된 PC라면 `python -m ipykernel install --user --name ai-fashion --display-name "AI Fashion"`을 실행하고 Notebook 커널을 **AI Fashion**으로 선택합니다.
4. `main.ipynb`를 열고 맨 위의 **팀원별 로컬 경로 설정** 셀을 수정합니다.
5. 사용 조건을 입력한 뒤 Notebook을 위에서 아래로 실행합니다.

### 로컬 경로 설정

Notebook에서는 다음 경로 값을 수정하면 됩니다.

```python
PROJECT_DIR_INPUT = r''
IMAGE_PATH_INPUT = r'data/input_person.jpg'
DATA_DIR_INPUT = r''
RULES_PATH_INPUT = r'FASHION_RULES_MASTER.md'
ATTRIBUTE_HEADS_PATH_INPUT = r'models/fashion_attribute_heads.pt'
OUTPUT_DIR_INPUT = r''
FONT_PATH_INPUT = r''
```

- 빈 `PROJECT_DIR_INPUT`은 현재 작업 폴더와 그 아래 `ai_fashion_recommender`를 자동 탐색합니다.
- 상대경로는 프로젝트 폴더를 기준으로 처리하므로 팀 공유에는 `data/input_person.jpg` 같은 형식을 권장합니다.
- 프로젝트를 자동으로 찾지 못할 때만 `PROJECT_DIR_INPUT`에 각자 프로젝트 절대경로를 입력합니다.
- `DATA_DIR_INPUT`과 `OUTPUT_DIR_INPUT`은 비워두면 각각 프로젝트 내부 `data`, `outputs`를 사용합니다.
- `RULES_PATH_INPUT`은 추천 엔진이 직접 읽는 패션 규칙 Markdown입니다. 문서의 `## R-...` 형식만 규칙으로 인식됩니다.
- `ATTRIBUTE_HEADS_PATH_INPUT`에 학습 체크포인트가 있으면 세부 카테고리·소매·기장·넥라인·칼라·핏·패턴·소재·디테일 분류를 사용하고, 없으면 기존 FashionSigLIP 제로샷 분석으로 자동 대체합니다.
- 통합 규칙 50개 중 43개가 구현되어 있습니다. 이 중 31개는 후보 필터·순위 점수에, 나머지는 분석 신뢰도 안전장치와 신발·액세서리 스타일링 안내에 사용됩니다. 상세 날씨와 보유 옷처럼 선택 입력이 필요한 규칙은 값이 있을 때만 실제 추천에 적용됩니다.
- 한글이 깨지는 환경에서는 `FONT_PATH_INPUT`에 설치된 한글 `.ttf` 또는 `.ttc` 파일을 지정합니다.

Notebook 밖에서 모듈만 사용할 때는 `FASHION_DATA_DIR`, `FASHION_OUTPUT_DIR`, `FASHION_FONT_PATH`, `FASHION_ATTRIBUTE_HEADS_PATH` 환경변수로 같은 경로를 변경할 수 있습니다. 별도 설정이 없으면 모든 경로는 `config.py`가 있는 프로젝트 폴더를 기준으로 결정됩니다.

기본 설정은 FASHN Human Parser와 FashionSigLIP을 모두 사용합니다. 첫 실행에서는 체크포인트를 내려받기 때문에 시간이 오래 걸릴 수 있고, 이후에는 로컬 캐시를 사용합니다. CPU에서도 실행할 수 있지만 FashionSigLIP 분석은 수십 초가 걸릴 수 있습니다.

FASHN 결과가 있어야 상·하의 영역과 옷 길이를 분석합니다. 포즈 기반 대체 마스크는 모델 연결 문제를 확인하기 위한 디버깅 수단이며 정식 결과로 취급하지 않습니다.

## FashionSigLIP 다중 속성 헤드 학습

`train_fashion_attribute_heads.ipynb`를 위에서부터 실행하면 다음 순서로 학습합니다.

1. 제공된 Fashionpedia + Fashion200K 통합 CSV를 사용하거나 같은 형식의 자체 CSV를 불러옵니다.
2. 고정된 FashionSigLIP으로 의류 이미지를 한 번만 임베딩하고 캐시합니다.
3. 카테고리·소매·기장·넥라인·칼라·핏·실루엣은 단일 분류, 패턴·소재·디테일은 복수 분류 헤드로 학습합니다.
4. 검증 세트 정확도와 micro-F1을 계산하고 복수 분류 임계값을 검증 데이터에서 선택합니다.
5. 결과를 `models/fashion_attribute_heads.pt`에 저장하면 `main.ipynb`가 자동으로 사용합니다.

빈 라벨은 `없음`이 아니라 **미주석**으로 처리해 해당 속성의 손실 계산에서 제외합니다. 학습 표본이 5장 미만인 라벨은 추론 시 자동으로 차단합니다. 현재 데이터는 Fashionpedia의 전문가 세부 속성 주석을 중심으로 하고, Fashion200K의 상품 분류 메타데이터로 셔츠·블라우스·폴로 셔츠·시각적 소재와 하의 세부 종류를 보완했습니다. [FashionSigLIP 모델 카드](https://huggingface.co/Marqo/marqo-fashionSigLIP), [Fashionpedia 공식 저장소](https://github.com/cvdfoundation/fashionpedia), [Fashion200K 데이터셋 카드](https://huggingface.co/datasets/Marqo/fashion200k)

Fashionpedia 이미지가 `이미지루트/train`, `이미지루트/val`처럼 나뉘어 있으면 변환 Notebook의 `TRAIN_IMAGE_PREFIX`, `VAL_IMAGE_PREFIX`에 폴더명을 입력합니다. 한 폴더에 합쳐져 있으면 두 값을 비워 둡니다.

현재 학습 스키마는 17개 헤드, 총 124개 라벨입니다. 하의는 한 개의 이름으로 억지로 합치지 않고 네 축으로 나눕니다. 종류는 `슬랙스·치노 팬츠·청바지·카고 팬츠·조거/스웨트팬츠·트랙팬츠·레깅스·하렘 팬츠·요가 팬츠·세일러 팬츠` 10종, 다리 모양은 `스키니·스트레이트·테이퍼드·페그·부츠컷·플레어·와이드·팔라초` 8종, 바지 전용 기장은 `카프리/7부·크롭/앵클·풀렝스` 3종입니다. 5포켓·카고 포켓·플리츠/턱·드로스트링·밴딩 허리·사이드 스트라이프·밑단 커프·스터럽도 복수 디테일로 분류합니다. 따라서 `팬츠 → 카고 팬츠 + 와이드 + 풀렝스 + 카고 포켓`처럼 조합됩니다.

현재 체크포인트는 의류 crop 5,984개(학습 4,789 / 검증 1,195)로 실제 학습되어 있습니다. 통합 검증의 전체 정확도는 의류 대분류 87.0%, 하의 종류 70.6%, 바지 다리 모양 57.0%, 바지 전용 기장 74.3%이고, 하의 구조 디테일 micro-F1은 88.1%입니다. 검증셋으로 신뢰도를 보정한 결과만 출력할 때 하의 종류 정확도는 80.3%(응답률 72.6%), 다리 모양은 85.0%(응답률 32.1%), 바지 기장은 81.9%(응답률 72.1%)입니다. 같은 공개 데이터 출처의 약지도 분할 결과이므로 실제 사용자 전신사진 성능을 보장하지 않으며, 신뢰도가 낮으면 세부 결과를 `분석 보류`로 남깁니다.

임베딩 캐시는 배치마다 중간 저장되므로 CPU에서 시간이 오래 걸려도 마지막 완료 배치부터 재개합니다. `cache --reuse-caches ...` 옵션을 주면 같은 이미지의 기존 FashionSigLIP 특징도 다시 계산하지 않습니다. 헤드는 속성별 검증 손실이 가장 낮았던 시점을 각각 저장합니다.

학습 체크포인트를 찾지 못하면 zero-shot fallback이 상의·원피스 14종과 하의 5종을 비교합니다. 다만 이 값은 전용 속성 헤드보다 거친 상대점수이므로 임시 분석으로 취급합니다.

라벨 스키마가 변경됐으므로 이전 스키마로 만든 `fashion_attribute_heads.pt`는 재학습해야 합니다. 로더는 라벨 순서가 다른 체크포인트를 오류로 차단합니다.

## DeepFashion으로 분석기 평가하기

DeepFashion 이미지는 프로젝트에 포함하지 않습니다. 공식 사용 동의 절차를 거쳐 `DeepFashion-MultiModal`을 받은 뒤 `deepfashion_evaluation.ipynb`의 경로 셀에서 다음 파일 위치를 지정합니다.

- 이미지 폴더
- shape 라벨 파일
- fabric 라벨 파일
- color/pattern 라벨 파일

평가 Notebook은 소매 길이, 하의 길이, 패턴, 소재, 네크라인의 응답률, 선택 정확도, 전체 정확도, macro F1과 대표 오답을 `outputs/deepfashion_evaluation.json`에 저장합니다. CPU에서는 `MAX_SAMPLES=20`처럼 작은 수로 먼저 확인하는 것을 권장합니다.

## 현재 한계

- 사진 기반 체형 비율은 실제 신체 치수가 아닙니다.
- 패션 규칙 MD는 실제 엔진에 연결되어 있지만 아직 전문가 합의와 사용자 실험 전의 후보 규칙입니다.
- 현재 추천 점수는 문서의 100점 중 기본 80점을 사용하고 보유 옷을 입력하면 활용도 5점이 추가됩니다. 실측 사이즈 15점은 사용자·상품 치수 데이터가 없어 제외합니다.
- 미지원 규칙 7개는 실측 사이즈, 레이어 밑단, 추천 상품의 색 면적, 작은 액세서리 상품, 트렌드 데이터가 필요한 규칙이며 Notebook 실행 시 ID를 표시합니다.
- 상품 CSV는 추천 로직 검증용이며 실제 쇼핑몰 상품이 아닙니다.
- 현재 출력은 실제 가상 피팅이 아니라 명시적으로 표시된 추천 보드입니다.
- IDM-VTON은 비상업 라이선스이므로 상용화 전 별도 모델·라이선스 검토가 필요합니다.
- FASHN Human Parser는 NVIDIA SegFormer 라이선스, FashionSigLIP 체크포인트는 Apache-2.0 조건을 각각 확인해야 합니다.
- DeepFashion 및 DeepFashion-MultiModal은 비상업 연구 전용이며 원본과 파생 데이터의 재배포가 제한됩니다.
- 다중 속성 헤드는 실제 공개 데이터로 학습했지만 클래스 불균형이 큽니다. 세일러 칼라·테이퍼드핏·디테일 없음과 표본 5장 미만 라벨은 현재 체크포인트가 출력하지 않습니다.
- Fashion200K 보완 라벨은 사람이 사진을 다시 검수한 정답이 아니라 상품 분류 메타데이터에서 만든 약지도 라벨입니다. 한국 사용자 전신사진으로 별도 외부 검증과 미세조정이 필요합니다.
- 슬랙스와 치노처럼 외형이 비슷한 하의는 원단·주름·허리 구조가 가려지면 사진만으로 구분하기 어렵습니다. 종류·다리 모양·기장·디테일을 서로 다른 속성으로 해석해야 합니다.
- 조드퍼·카펜터·스코트처럼 공개 표본이 부족하거나 점프수트처럼 전신 카테고리에 해당하는 이름은 이번 바지 세부 헤드에서 억지로 출력하지 않습니다.
