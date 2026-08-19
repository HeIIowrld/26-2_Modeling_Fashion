# `main.ipynb` 구조 설명

## 전체 흐름

`main.ipynb`는 각 기능을 직접 구현하는 파일이 아니라, 같은 폴더의 Python 모듈을 순서대로 불러 실행하는 메인 파일이다.

```text
로컬 경로 설정
→ 사용자 조건 입력
→ 사진 품질 검사
→ 체형·자세 분석
→ 현재 착장 분석
→ 코디 후보 생성 및 순위 결정
→ 결과 이미지 생성
→ 최종 결과 요약 및 피드백 저장
```

## 셀별 구조

### 0. 팀원별 로컬 경로 설정

팀원마다 다른 폴더 구조를 사용할 수 있도록 프로젝트, 입력 사진, 데이터, 패션 규칙, 출력, 글꼴 경로를 설정한다.

```python
PROJECT_DIR_INPUT = r''
IMAGE_PATH_INPUT = r'data/input_person.jpg'
DATA_DIR_INPUT = r''
RULES_PATH_INPUT = r'FASHION_RULES_MASTER.md'
ATTRIBUTE_HEADS_PATH_INPUT = r'models/fashion_attribute_heads.pt'
OUTPUT_DIR_INPUT = r''
FONT_PATH_INPUT = r''
```

`PROJECT_DIR_INPUT`, `DATA_DIR_INPUT`, `OUTPUT_DIR_INPUT`을 비워두면 프로젝트 내부 경로를 자동으로 사용한다. 이후 단계에 필요한 클래스도 이 셀에서 불러온다.

### 1. 사용자 조건과 모델 설정

추천에 사용할 목적, 스타일, 예산, 변경 범위 등을 입력한다.

- 코디 목적: 데일리, 데이트, 출근, 여행 등
- 원하는 스타일
- 예산
- 변경 범위: 현재 유지, 상의만, 하의만, 전체 변경
- 키·체중, 계절, 원하는 실루엣 목표

또한 FASHN 의류 파서, FashionSigLIP, 학습된 다중 속성 헤드, VTON 사용 여부를 설정한다. 속성 헤드 체크포인트가 없으면 기존 FashionSigLIP 제로샷 분석으로 자동 대체하며, 현재 VTON은 실제 모델이 연결되지 않아 기본값이 `False`다.

### 2. 입력 사진 품질 검사

`quality_checker.py`와 `pose_analyzer.py`를 사용해 전신이 충분히 보이는지, 이미지가 너무 작거나 흐리지 않은지 확인한다. MediaPipe 포즈 추론은 이 단계에서 한 번만 실행하며 이후 단계에서 결과를 재사용한다.

기준을 통과하지 못하면 문제 항목을 출력하고 실행을 중단한다.

### 3. MediaPipe 체형·자세 분석

검출된 어깨, 골반, 무릎, 발목 등의 랜드마크를 시각화하고 다음 값을 계산한다.

- 어깨·골반 비율
- 상체·하체 비율
- 다리 길이 비율
- 상체 강조형, 하체 강조형, 균형형 참고 분류와 분류 신뢰도
- 정면에 가까운 자세인지 여부

사진에서 구한 값이므로 실제 신체 치수가 아니라 코디 추천을 위한 상대적인 참고값이다. 자세나 관절 가시성이 부족하거나 분류 경계에 가까우면 `분석 불확실`로 보류한다. 사용자가 실루엣 목표를 고르지 않으면 이 체형 라벨은 추천 점수에 사용하지 않는다.

### 4. 의류 분리와 현재 착장 분석

FASHN Human Parser로 상의와 하의 영역을 분리하고 스카프·벨트·가방 같은 액세서리는 별도 마스크로 둔다. 관절 위치와 분할 마스크를 함께 사용해 옷의 길이와 상·하의 핏을 추정한다. 학습 체크포인트가 있으면 FashionSigLIP 이미지 특징 위의 17개 속성 헤드가 총 124개 라벨로 종류, 하의 세부 종류, 소매 형태, 기장, 넥라인, 칼라, 핏, 실루엣, 패턴, 소재, 디테일을 분류한다. 하의는 팬츠·청바지·쇼츠·스커트 대분류와 별도로 종류, 다리 모양, 바지 전용 기장, 구조 디테일을 각각 출력한다. 예를 들어 `카고 팬츠 + 와이드 + 풀렝스 + 카고 포켓`처럼 여러 특징을 동시에 표현할 수 있다. 체크포인트가 없거나 해당 속성의 신뢰도가 부족하면 기존 제로샷 결과와 마스크 측정값을 사용한다.

주요 출력값은 다음과 같다.

- 상의·하의 대분류, 하의 종류·다리 모양·기장·구조 디테일과 CIELAB 기반 주요 색상·면적 비율
- 소매, 상의, 하의 길이
- 핏과 실루엣
- 스타일, 패턴, 소재, 넥라인, 칼라, 소매 형태, 디테일
- 속성별 결과 출처와 신뢰도

분할 결과 이미지는 `outputs/fashn_segmentation.jpg`에 저장된다.

### 5. 코디 후보 탐색과 순위 결정

`products.csv`에서 재고가 있는 상품을 불러온 뒤 예산, 목적, 계절 조건에 맞지 않는 상품을 먼저 제외한다. `FASHION_RULES_MASTER.md`에 해당 R-* 규칙이 있을 때만 아래 점수를 적용한다.

| 평가 항목 | 비중 |
|---|---:|
| 목적·TPO·격식 적합도 | 20% |
| 날씨·활동 적합도 | 12% |
| 실루엣 조화 | 15% |
| 색상 조화 | 15% |
| 패턴·소재·복잡도 | 10% |
| 사용자 취향 | 8% |
| 보유 옷 활용도 | 5% (보유 옷 입력 시) |

기본 입력에서는 규칙 문서의 총 100점 중 80점을 사용하고, 보유 옷 목록이 있으면 활용도 5점을 추가한다. 목적별로 세부 비중을 조정한 뒤 현재 계산 가능한 범위 안에서 100점으로 정규화한다. 통합 규칙 50개 중 43개가 코드에 연결되어 있으며, 31개는 후보 필터·순위 점수, 나머지는 신뢰도 안전장치·분석 파이프라인·신발과 액세서리 안내에 사용한다. 실측 사이즈, 레이어 밑단, 상품 이미지 색 면적, 작은 액세서리 상품과 추세 데이터가 필요한 7개 규칙은 미지원 사유를 표시한다.

### 6. 결과 이미지 생성

가장 높은 점수를 받은 코디를 결과 이미지로 만든다. 현재 `USE_VTON=False`일 때는 실제 가상 피팅이 아니라 입력 사진과 추천 정보를 조합한 미리보기 보드를 생성한다.

결과는 `outputs/recommendation_preview.jpg`에 저장된다. 실제 착장 합성을 사용하려면 추후 `virtual_tryon.py`에 VTON 모델을 연결해야 한다.

### 7. 피드백 저장과 최종 요약

사진 품질, 체형, 의류 분석 결과, 최고 추천 점수, 결과 이미지 경로를 하나의 딕셔너리로 정리한다. `SAVE_EXAMPLE_FEEDBACK=True`로 바꾸면 사용자의 반응을 `outputs/feedback.jsonl`에 저장할 수 있다.

## 주요 파일의 역할

| 파일 | 역할 |
|---|---|
| `main.ipynb` | 전체 파이프라인 실행 |
| `pose_analyzer.py` | MediaPipe 체형·자세 분석 |
| `quality_checker.py` | 입력 사진 품질 검사 |
| `clothing_parser.py` | FASHN 의류 영역 분할 |
| `garment_attribute_analyzer.py` | 소매·상의·하의 길이와 핏 추정 |
| `fashion_model.py` | FashionSigLIP 추론 |
| `fashion_attribute_schema.py` | 속성별 라벨 체계 정의 |
| `fashion_attribute_dataset.py` | 학습 CSV·Fashionpedia 주석 변환 |
| `fashion_attribute_model.py` | 속성별 분류 헤드와 추론 |
| `fashion_attribute_training.py` | 임베딩 캐시·학습·평가 |
| `train_fashion_attribute_heads.ipynb` | 다중 속성 헤드 단계별 학습 |
| `outfit_analyzer.py` | 현재 착장 분석 결과 통합 |
| `fashion_rules.py` | 패션 규칙 Markdown 로딩 |
| `product_catalog.py` | 상품 CSV 로딩 |
| `recommendation_engine.py` | 후보 생성, 점수 계산, 순위 결정 |
| `virtual_tryon.py` | 결과 보드 생성 및 향후 VTON 연결 지점 |
| `feedback_store.py` | 사용자 피드백 저장 |

## 팀원이 실행하는 방법

1. ZIP 파일을 원하는 폴더에 압축 해제한다.
2. Python 3.11 환경을 권장한다.
3. 터미널에서 `pip install -r requirements.txt`를 실행한다.
4. Python이 여러 개면 `python -m ipykernel install --user --name ai-fashion --display-name "AI Fashion"`으로 현재 환경을 등록하고 해당 커널을 선택한다.
5. 자신의 정면 전신사진을 `data/input_person.jpg`로 넣거나 `IMAGE_PATH_INPUT`을 수정한다.
6. Jupyter에서 `main.ipynb`를 열고 위에서부터 순서대로 실행한다.

첫 실행에서는 FASHN Parser와 FashionSigLIP 체크포인트를 내려받기 때문에 인터넷 연결이 필요하며 시간이 더 걸릴 수 있다. GPU가 있으면 FashionSigLIP이 자동으로 CUDA를 사용하고, 없으면 CPU로 실행된다.

## 현재 단계에서 알아둘 점

- 체형 값은 사진 기반의 상대 비율이며 실제 신체 치수가 아니다.
- 추천 규칙과 상품 목록은 MVP 검증용 데이터다.
- 현재 결과 이미지는 실제 가상 피팅이 아니라 추천 보드다.
- 입력 전신사진과 모델 체크포인트는 배포 ZIP에 포함되지 않는다.
- DeepFashion 원본 데이터는 별도 이용 신청과 다운로드가 필요하다.
- `models/fashion_attribute_heads.pt`가 없으면 세부 속성 헤드는 비활성화되고 기존 제로샷 방식으로 실행된다.
