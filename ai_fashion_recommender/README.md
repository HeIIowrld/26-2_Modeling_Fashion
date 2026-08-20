# ai_fashion_recommender

전신사진 한 장으로 체형과 착장을 읽고, 규칙으로 상품을 고르고, 실제로 입혀본
합성 사진까지 만드는 코드. `main.ipynb`가 전체 흐름을 순서대로 보여주고
기능별 모듈은 `src/`에 있다.

## 구조

```
├── main.ipynb                  전체 파이프라인 단계별 실행
├── app.py                      Gradio 웹 앱
├── FASHION_RULES_MASTER.md     추천 엔진이 런타임에 읽는 R-* 규칙
├── src/                        런타임 모듈
├── scripts/                    수집·전처리·학습·검증 스크립트
├── docs/                       규칙 연구 노트, 설치·구조 문서
├── tests/                      단위·회귀 테스트
├── experiments/                지난 학습 실험 기록
├── data/                       카탈로그 CSV, 학습 주석, 규칙 JSON
├── models/                     학습된 속성 헤드 체크포인트
├── reports/                    실험 리포트
└── outputs/                    실행 산출물 (git 제외)
```

`src/`가 `sys.path`에 등록되므로 모듈끼리는 `from config import ...`처럼 평면 임포트를
쓴다. Notebook은 0번 셀에서, `app.py`와 `scripts/*.py`는 파일 상단에서 이 경로를 잡는다.
이미지 자산은 코드 밖 `../datasets/`에 있고 `config.py`의 `PEOPLE_DIR`,
`GARMENT_RAW_DIR`, `GARMENT_CLEAN_DIR`로만 접근한다.

### src/

| 모듈 | 역할 |
| --- | --- |
| `config.py` | 경로·임계값. `PROJECT_DIR`은 `src/`의 상위 |
| `schemas.py` | 사용자·분석·상품·추천 데이터 구조 |
| `pose_analyzer.py` | MediaPipe Pose 기반 체형·자세 비율 |
| `clothing_parser.py` | FASHN Human Parser로 의류 픽셀 마스크 생성 |
| `garment_attribute_analyzer.py` | 마스크 끝점과 관절로 소매·기장·핏 측정 |
| `outfit_analyzer.py` | 색상과 속성 분석 결과 통합 |
| `quality_checker.py` | 입력·합성 결과 품질 검사 |
| `fashion_model.py` | FashionSigLIP + 학습된 속성 헤드 추론 |
| `fashion_prompts.py` | zero-shot용 패턴·소재·네크라인 후보 |
| `fashion_attribute_schema.py` | 속성별 라벨과 단일·복수 분류 정의 |
| `fashion_attribute_model.py` | 고정 특징 위의 속성별 분류 헤드 |
| `fashion_attribute_dataset.py` | 학습 CSV 로딩, Fashionpedia 주석 변환 |
| `fashion_attribute_training.py` | 특징 캐시, 학습, 평가, 임계값 선택 |
| `fashion_rules.py` | 규칙 문서에서 R-* ID와 메타데이터 로딩 |
| `recommendation_engine.py` | 규칙 기반 필터·점수·설명 생성 |
| `product_catalog.py` | 카탈로그 로딩, 카테고리·성별 필터 |
| `virtual_tryon.py` | VTON 인터페이스와 비합성 추천 보드 |
| `catvton_tryon.py` | CatVTON 어댑터 (FASHN 마스크 사용) |
| `deepfashion_dataset.py` | DeepFashion 라벨 로딩과 정확도 평가 |
| `feedback_store.py` | 피드백 JSONL 저장 |

### scripts/

- `test_tryon.py` — 수집 사진으로 추천~합성 전후 비교
- `musinsa_crawler.py` — 상품 메타데이터·전면샷 수집 (연구용 소규모)
- `prepare_fashionpedia_seed.py` — Fashionpedia 주석·이미지 추출
- `prepare_fashion200k_supplement.py` — 셔츠·블라우스·폴로·소재 보완 샘플
- `prepare_fashion200k_bottoms.py` — 하의 종류·다리 모양·기장·디테일 보완 샘플
- `train_fashion_attribute_heads.py` — 속성 헤드 학습 CLI

## 실행

```bash
pip install -r requirements.txt
python scripts/test_tryon.py --vton --count 2    # 수집 사진으로 바로 확인
python app.py                                    # 웹 앱 (localhost:7860)
python app.py --light                            # 무거운 모델 없이 UI 흐름만
```

Notebook을 쓰려면 `main.ipynb`를 열고 맨 위 경로 셀만 고치면 된다.
Python이 여러 개 깔린 PC라면 커널을 먼저 등록해 둔다.

```bash
python -m ipykernel install --user --name ai-fashion --display-name "AI Fashion"
```

### 경로 설정

```python
PROJECT_DIR_INPUT = r''                                        # 비우면 자동 탐색
IMAGE_PATH_INPUT = r'data/input_person.jpg'
DATA_DIR_INPUT = r''
RULES_PATH_INPUT = r'FASHION_RULES_MASTER.md'
ATTRIBUTE_HEADS_PATH_INPUT = r'models/fashion_attribute_heads.pt'
OUTPUT_DIR_INPUT = r''
FONT_PATH_INPUT = r''                                          # 한글 깨질 때만
```

상대경로는 프로젝트 폴더 기준이라 팀 공유에는 `data/input_person.jpg` 형태가 편하다.
자동 탐색에 실패할 때만 `PROJECT_DIR_INPUT`에 절대경로를 넣는다.
규칙 문서는 `## R-...` 형식만 규칙으로 인식한다. 속성 체크포인트가 없으면
zero-shot 분석으로 자동 대체된다.

Notebook 밖에서는 `FASHION_DATA_DIR`, `FASHION_OUTPUT_DIR`, `FASHION_DATASETS_DIR`,
`FASHION_FONT_PATH`, `FASHION_ATTRIBUTE_HEADS_PATH` 환경변수로 같은 경로를 바꿀 수 있다.

첫 실행은 체크포인트를 받느라 오래 걸리고 이후에는 로컬 캐시를 쓴다. CPU에서도 돌지만
FashionSigLIP 분석에 수십 초가 걸린다. FASHN 결과가 있어야 상·하의 영역과 기장을
분석한다. 포즈 기반 대체 마스크는 모델 연결을 확인하는 디버깅용이고 정식 결과가 아니다.

## 가상 피팅 (CatVTON)

`python app.py --vton` 또는 `main.ipynb`에서 `USE_VTON = True`로 바꾸면 추천 보드 대신
실제로 옷을 입힌 합성 사진이 나온다.

- 상위 폴더에 저장소를 클론해 둔다.
  `git clone https://github.com/Zheng-Chong/CatVTON.git third_party/CatVTON`
- 마스크는 원본의 AutoMasker(detectron2 필요) 대신 FASHN 마스크를 쓴다.
  덕분에 Windows에서도 돈다.
- 첫 실행에 HuggingFace에서 약 5GB를 받는다.
- CPU는 장당 수 분 이상 걸린다. GPU 권장.
- 가중치는 CC BY-NC-SA 4.0(비상업)이다.

AMD RX 7000/9000 시리즈는 Windows용 ROCm PyTorch 프리뷰로 가속할 수 있다.
설치 절차는 [docs/AMD_ROCM_SETUP.md](docs/AMD_ROCM_SETUP.md)에 정리해 뒀다.

## 합성 품질

같은 사람·마스크·시드에서 상품만 바꾼 통제 실험(2026-08-20)으로 확인한 내용이다.
근거 이미지와 상세 기록은 [reports/vton_quality/](reports/vton_quality/)에 있다.

### 원래 옷보다 짧은 옷은 깨진다 (원인 확인됨)

마스크는 지금 입고 있는 옷 기준으로 만들어진다. 더 짧은 옷을 넣으면 남는 마스크
영역(종아리·팔)을 모델이 맨살이 아니라 옷 비슷한 것으로 채운다.

| 기장 차이 | 결과 |
| --- | --- |
| 긴바지 → 미디·7부 (gap 1) | 정상 |
| 긴바지 → 쇼츠·미니 (gap 3) | 다리 전체가 니트 텍스처로 덮임 |

소매도 같다. 긴팔 마스크에 반팔 상품을 넣으면 긴팔로 그려진다.
`bottom_length_gap` / `sleeve_length_gap`이 2단계 이상 차이를 경고한다.
gap 1은 정상, gap 3은 실패를 실측했고 gap 2는 미검증이라 보수적으로 포함했다.

### 시스루 렌더링 (원인 미확정)

특정 레퍼런스(MS6797005)에서 옷이 반투명하게 나온다. 상의 레퍼런스만 5종 교체한 결과:

| 레퍼런스 | coverage | contrast | 색 | 결과 |
| --- | --- | --- | --- | --- |
| MS6677115 | 0.50 | 208 | 갈색 | 불투명 |
| MS6797005 | 0.15 | 56 | 흰색 | 시스루 |
| MS6843814 | 0.06 | 230 | 검정 | 불투명 |
| MS6162377 | 1.47 | 47 | 흰색 | 불투명·충실 |
| MS6529592 | 0.30 | 228 | 검정 | 불투명 |

기각한 가설이 다섯이다. 마스크 구멍(양쪽 0%), coverage 단독(0.06인데 정상),
contrast 단독(흰옷인데 정상), 흰 배경에 묻힘(회색 매트로도 시스루 유지),
레퍼런스 자체(같은 레퍼런스가 다른 사람에게는 최고 품질). 시드를 바꿔도 그대로다.
즉 입력 지표만으로는 예측되지 않으며, `min_reference_coverage`(0.25) 경고는
품질 참고용일 뿐 시스루 예측기로 검증된 값이 아니다.

### 점검 결과 쓰기

`generate()`에 `context={"outfit": ..., "classifier": ...}`를 같이 넘기면 위 점검이 돌고
사유가 `tryon.last_warnings`에 쌓인다. 웹 앱은 분석 패널에, Notebook과
`scripts/test_tryon.py`는 결과 아래에 찍는다. 경고가 붙은 결과는 합성이 깨졌을 수
있으니 그대로 믿으면 안 된다.

### 아직 못 고친 것

- FASHN이 옷 위의 가방끈을 `top`으로 분류하는 경우가 있어 세그멘테이션으로 못 지운다.
  가방·머리카락·팔로 분류되는 가림 요소만 정제 단계에서 뺀다.
- 마스크 모양 프라이어는 CatVTON 구조상 우회할 수 없다. 기장 문제는 감지·안내까지만 된다.
- 세로로 긴 사진은 `pad_to_aspect`로 여백을 덧대 잘림을 막는다. 새 소스를 넣을 때는
  종횡비부터 확인할 것. 인스타 수집본은 중앙값이 약 1:2였다.

## 속성 헤드 학습

`train_fashion_attribute_heads.ipynb`를 위에서부터 실행하면 된다.

1. Fashionpedia + Fashion200K 통합 CSV를 쓰거나 같은 형식의 자체 CSV를 넣는다.
2. 고정된 FashionSigLIP으로 의류 이미지를 한 번만 임베딩해 캐시한다.
3. 카테고리·소매·기장·넥라인·칼라·핏·실루엣은 단일 분류, 패턴·소재·디테일은 복수 분류로 학습한다.
4. 검증 정확도와 micro-F1을 계산하고 복수 분류 임계값을 검증 데이터에서 고른다.
5. `models/fashion_attribute_heads.pt`에 저장하면 `main.ipynb`가 자동으로 쓴다.

빈 라벨은 "없음"이 아니라 미주석으로 처리해 손실 계산에서 뺀다. 학습 표본이
5장 미만인 라벨은 추론에서 자동 차단한다. 데이터는 Fashionpedia의 전문가 주석이
중심이고, Fashion200K 상품 메타데이터로 셔츠·블라우스·폴로와 하의 세부 종류를 채웠다.
([FashionSigLIP](https://huggingface.co/Marqo/marqo-fashionSigLIP) ·
[Fashionpedia](https://github.com/cvdfoundation/fashionpedia) ·
[Fashion200K](https://huggingface.co/datasets/Marqo/fashion200k))

스키마는 17개 헤드, 124개 라벨이다. 하의는 한 이름으로 뭉치지 않고 네 축으로 나눈다.
종류 10종(슬랙스·치노·청바지·카고·조거·트랙·레깅스·하렘·요가·세일러), 다리 모양 8종
(스키니·스트레이트·테이퍼드·페그·부츠컷·플레어·와이드·팔라초), 바지 기장 3종
(카프리/7부·크롭/앵클·풀렝스), 그리고 구조 디테일을 복수로 붙인다. 그래서
`팬츠 → 카고 팬츠 + 와이드 + 풀렝스 + 카고 포켓`처럼 조합된다.

현재 체크포인트는 의류 crop 5,984개(학습 4,789 / 검증 1,195)로 학습했다.
전체 정확도는 대분류 87.0%, 하의 종류 70.6%, 다리 모양 57.0%, 바지 기장 74.3%,
하의 디테일 micro-F1 88.1%. 신뢰도로 거른 결과만 보면 하의 종류 80.3%(응답률 72.6%),
다리 모양 85.0%(32.1%), 바지 기장 81.9%(72.1%)다. 같은 출처의 약지도 분할이라
실제 사용자 사진 성능을 보장하지 않고, 신뢰도가 낮으면 `분석 보류`로 남긴다.

임베딩 캐시는 배치마다 저장되므로 CPU에서 오래 걸려도 마지막 배치부터 재개된다.
`cache --reuse-caches`를 주면 기존 특징도 재계산하지 않는다. 라벨 스키마가 바뀌면
이전 체크포인트는 재학습해야 한다. 로더가 라벨 순서가 다른 체크포인트를 막는다.

## DeepFashion 평가

DeepFashion 이미지는 저장소에 없다. 공식 동의 절차를 거쳐 `DeepFashion-MultiModal`을
받은 뒤 `deepfashion_evaluation.ipynb`의 경로 셀에서 이미지 폴더와 shape / fabric /
color-pattern 라벨 파일 위치를 지정한다. 소매 길이, 하의 길이, 패턴, 소재, 네크라인의
응답률·선택 정확도·전체 정확도·macro F1과 대표 오답을
`outputs/deepfashion_evaluation.json`에 남긴다. CPU라면 `MAX_SAMPLES=20`처럼
작게 잡고 먼저 확인하는 편이 낫다.

## 한계

- 사진에서 계산한 체형 비율은 실제 신체 치수가 아니다.
- 패션 규칙은 엔진에 연결돼 있지만 전문가 합의와 사용자 실험 전의 후보 규칙이다.
- 추천 점수는 문서 100점 중 기본 80점을 쓴다. 보유 옷을 넣으면 활용도 5점이 붙고,
  실측 사이즈 15점은 치수 데이터가 없어 빠진다. 미지원 규칙 7개는 실행 시 ID를 보여준다.
- 속성 헤드는 클래스 불균형이 크다. 세일러 칼라·테이퍼드핏 같은 희소 라벨과
  표본 5장 미만 라벨은 출력하지 않는다.
- Fashion200K 보완 라벨은 사람이 재검수한 정답이 아니라 상품 메타데이터에서 만든
  약지도 라벨이다. 한국 사용자 사진으로 외부 검증과 미세조정이 필요하다.
- 슬랙스와 치노처럼 비슷한 하의는 원단·주름·허리가 가려지면 사진만으로 구분이 어렵다.
- 상품 CSV는 추천 로직 검증용이며 실시간 재고를 반영하지 않는다.
- 모델별 라이선스를 각각 확인해야 한다. FASHN Human Parser는 NVIDIA SegFormer,
  FashionSigLIP은 Apache-2.0, CatVTON은 CC BY-NC-SA 4.0, DeepFashion 계열은
  비상업 연구 전용이다.
