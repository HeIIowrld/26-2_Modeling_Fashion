# 0단계 — 현재 레포 이해하기

> 대상: `ai_fashion_recommender/`
> 정리 기준일: 2026-08-18
> 핵심 질문 3개: **어떤 모델을 쓰는가 / 어떤 학습 데이터를 쓰는가 / 어떤 평가지표를 쓰는가**

---

## 0. 30초 요약

| 질문 | 답 |
|---|---|
| 학습되는 모델 | **FashionSigLIP(고정) + 17개 속성 분류 헤드(MLP, 학습 대상)** |
| 학습 파라미터 | 백본은 **완전 freeze**, 헤드만 학습 (약 3.4M 파라미터) |
| 학습 데이터 | **Fashionpedia + Fashion200K**에서 만든 의류 crop **5,984장** (train 4,789 / val 1,195) |
| 라벨 체계 | 17개 속성(task) × 총 124개 라벨. 단일분류 13개 + 다중분류 4개 |
| 평가지표 | 단일분류 = **Accuracy** (+ 임계값 통과분 accepted accuracy / coverage), 다중분류 = **micro-F1**. 라벨별 precision/recall/F1도 저장 |
| 현재 성능 | 대분류 87.0%, 최약점 `pant_leg_shape` 57.0% |
| 저장 위치 | `models/fashion_attribute_heads.pt` + `models/fashion_attribute_heads.metrics.json` |

이 레포는 "추천 시스템 전체"지만, **학습·평가되는 ML 모델은 딱 하나**입니다 — 의류 속성 분류 헤드.
나머지(포즈, 파싱, 추천 규칙)는 학습이 없는 **사전학습 모델 + 규칙 엔진**입니다.

---

## 1. 어떤 모델을 사용하는가

### 1-1. 전체 파이프라인에 쓰이는 모델 4종

```
사용자 전신사진
   │
   ├─(A) MediaPipe Pose ─────────► 어깨/골반/다리 비율 (학습 없음, 사전학습)
   │        pose_analyzer.py
   │
   ├─(B) FASHN Human Parser ─────► 의류 종류별 픽셀 마스크 (학습 없음, 사전학습)
   │        clothing_parser.py        = NVIDIA SegFormer 기반
   │                │
   │                └─► garment_attribute_analyzer.py : 마스크 × 관절 → 소매/기장/핏 "측정"
   │
   ├─(C) FashionSigLIP ──────────► 이미지 임베딩 (★고정 = freeze)
   │        Marqo/marqo-fashionSigLIP
   │                │
   │                ▼
   │        ★ (D) 다중 속성 분류 헤드  ← ★★ 이 레포에서 유일하게 "학습"하는 모델 ★★
   │             fashion_attribute_model.py / models/fashion_attribute_heads.pt
   │
   ▼
recommendation_engine.py (규칙 기반, 학습 없음)
   + fashion_rules.py ← FASHION_RULES_MASTER.md 의 R-* 규칙 50개(43개 구현)
   ▼
추천 결과 + 설명 (virtual_tryon.py 는 실제 VTON 아님, 추천 보드)
```

### 1-2. ★ 학습 대상 모델의 정확한 구조

파일: [fashion_attribute_model.py](../fashion_attribute_model.py) `build_attribute_heads()`

```
FashionSigLIP 이미지 임베딩 (768차원, L2 정규화, requires_grad=False)
        │
        ├─ head["category"]        ┐
        ├─ head["lower_subtype"]   │  각 헤드가 완전히 동일한 구조:
        ├─ head["pant_leg_shape"]  │
        ├─ ...                     ├─  LayerNorm(768)
        └─ head["detail"]          │   → Linear(768 → 256)
           (총 17개 헤드)          │   → GELU
                                   │   → Dropout(0.3)
                                   ┘   → Linear(256 → 라벨 수)
```

- **멀티태스크 하드 파라미터 공유가 아님**: 17개 헤드가 백본 임베딩만 공유하고 서로 파라미터를 공유하지 않음 → 각 태스크가 독립적으로 학습됨(= 태스크 간 간섭 없음, 대신 데이터 적은 태스크는 도움도 못 받음)
- 백본이 freeze이므로 학습은 **임베딩 캐시(.pt) 위에서만** 돌아감 → CPU로도 수십 초 ~ 수 분
- 임베딩 캐시: `data/cache/fashion_attributes_train.pt`(16.7MB), `..._val.pt`(4.2MB)

### 1-3. 추론 시 안전장치 (중요 — 성능 해석에 영향)

`FashionAttributePredictor.predict_features()` 는 그냥 argmax를 내놓지 않습니다.

1. **label_support 마스킹**: 학습 표본이 `minimum_label_examples=5`장 미만인 라벨은 로짓을 `-inf`로 막음
   → 그래서 `컬러 블록`(train 1장), `O라인`(3장), `카모`(5장 경계) 등은 아예 출력되지 않음
2. **임계값 게이팅**: confidence < threshold 이면 라벨을 비우고 `accepted=False` → 사용자에게는 `분석 보류`
3. **다중분류 empty label 우선순위**: `무지`/`디테일 없음`이 실제 속성과 동시 선택되면 실제 속성만 남김
4. **mask 측정 vs 학습 헤드 융합**: `fuse_measured_and_learned()` 가 FASHN 마스크 측정값과 헤드 예측을 비교해
   일치 시 신뢰도 상향, 불일치 시 confidence가 0.15 이상 높은 쪽 채택

> 즉 이 모델의 실제 지표는 **"전체 정확도"와 "응답한 것만의 정확도(accepted accuracy)" 두 개**를 같이 봐야 합니다.

---

## 2. 어떤 학습 데이터(training set)를 사용하는가

### 2-1. 데이터 출처 3개

| 출처 | 행 수 | train / val | 라벨 성격 | 만드는 스크립트 |
|---|---|---|---|---|
| `fashionpedia_seed` | 2,032 | 1,625 / 407 | **전문가 세부 속성 주석** (강한 라벨) | `prepare_fashionpedia_seed.py` |
| `fashion200k_bottoms` | 3,132 | 2,508 / 624 | 상품명 파싱 → **약지도(weak) 라벨** | `prepare_fashion200k_bottoms.py` |
| `fashion200k_supplement` | 820 | 656 / 164 | 상품 카테고리 메타데이터 → **약지도 라벨** | `prepare_fashion200k_supplement.py` |
| **합계** | **5,984** | **4,789 / 1,195** | | |

- 통합 CSV: `data/fashion_attribute_annotations.csv`
- 이미지: `data/{출처}/images/` (실제 파일 5,363장, 총 172MB)
- **한 행 = 이미지 1장 + bbox 1개 = 의류 crop 1개** (한 사진에 상의/하의가 있으면 2행)
- Fashionpedia는 `convert_fashionpedia_instances()`가 영문 ontology(`ATTRIBUTE_KEYWORDS`)를 한국어 라벨로 매핑

### 2-2. CSV 스키마

```
image_path, split, bbox_x, bbox_y, bbox_w, bbox_h,
category, lower_subtype, pant_leg_shape, pant_length, lower_detail,
sleeve_length, sleeve_shape, upper_length, lower_length,
neckline, collar, upper_fit, lower_fit, silhouette,
pattern, material, detail
```

### 2-3. ★ 가장 중요한 설계 결정: "빈 값 = 없음"이 아니라 "빈 값 = 미주석"

```python
# fashion_attribute_dataset.py encode_record_targets()
valid[task_name] = bool(values)   # 라벨이 비어 있으면 valid=False
```

`_multitask_loss()`가 `valid` 마스크로 **손실 계산에서 아예 제외**합니다.

그래서 **태스크별 실제 학습/검증 표본 수가 크게 다릅니다**:

| task | val 표본 | 비고 |
|---|---|---|
| category | 1,195 | 전 행에 있음 |
| lower_fit | 442 | |
| lower_length | 422 | |
| pant_leg_shape | 374 | |
| pattern | 389 | |
| detail | 262 | |
| lower_subtype | 252 | |
| pant_length | 222 | |
| sleeve_length | 214 | |
| upper_fit | 178 | |
| material | 156 | |
| upper_length | 132 | |
| sleeve_shape | 122 | |
| lower_detail | 110 | ← 표본 최소권 |
| silhouette | 101 | |
| collar | 87 | |
| neckline | 85 | ← 표본 최소권 |

→ **neckline/collar/silhouette/lower_detail은 val 표본 100개 안팎**이라 정확도 1%가 표본 1장 수준입니다. 지표 해석 시 반드시 감안해야 합니다.

### 2-4. 심각한 클래스 불균형

`category`의 경우 train 4,789개 중 `팬츠` 2,578개(54%) — Fashion200K bottoms 보강 때문입니다.
반면 `후드티` 6장, `가디건` 11장, `점프수트` 16장, `니트` 17장.

대응 코드는 이미 있습니다:
- 단일분류: `_single_class_weights()` — **√역빈도** 가중치, mean 정규화 후 최대 10배로 clamp
- 다중분류: `_positive_weights()` — `pos_weight = 음성/양성`, 최대 20배로 clamp

### 2-5. 17개 속성 스키마 (총 124 라벨)

파일: [fashion_attribute_schema.py](../fashion_attribute_schema.py)

| task | 방식 | 라벨 | min_conf |
|---|---|---|---|
| `category` | 단일 | 티셔츠·폴로 셔츠·셔츠·블라우스·니트·가디건·후드티·재킷·블레이저·코트·베스트·탑·팬츠·청바지·쇼츠·스커트·원피스·점프수트 (18) | 0.45 |
| `lower_subtype` | 단일 | 슬랙스·치노·청바지·카고·조거·트랙·레깅스·하렘·요가·세일러 (10) | 0.48 |
| `pant_leg_shape` | 단일 | 스키니·스트레이트·테이퍼드·페그·부츠컷·플레어·와이드·팔라초 (8) | 0.45 |
| `pant_length` | 단일 | 카프리·7부 / 크롭·앵클 / 풀렝스 (3) | 0.48 |
| `lower_detail` | **다중** | 5포켓·카고포켓·플리츠·드로스트링·밴딩허리·사이드스트라이프·밑단커프·스터럽 (8) | 0.45 |
| `sleeve_length` | 단일 | 민소매·반팔·7부·긴팔 (4) | 0.50 |
| `sleeve_shape` | 단일 | 기본·퍼프·벌룬·벨·래글런·캡 (6) | 0.50 |
| `upper_length` | 단일 | 크롭·기본·롱 (3) | 0.50 |
| `lower_length` | 단일 | 쇼츠·미니 / 무릎 / 미디·7부 / 롱·긴바지 (4) | 0.50 |
| `neckline` | 단일 | 라운드·V·스퀘어·보트·오프숄더·홀터·터틀 (7) | 0.52 |
| `collar` | 단일 | 없음·폴로·셔츠·스탠드·라펠·피터팬·세일러 (7) | 0.52 |
| `upper_fit` | 단일 | 슬림·레귤러·여유·오버 (4) | 0.50 |
| `lower_fit` | 단일 | 슬림·스트레이트·테이퍼드·와이드·플레어 (5) | 0.50 |
| `silhouette` | 단일 | H·A·X·O·비대칭 (5) | 0.50 |
| `pattern` | **다중** | 무지·스트라이프·체크·도트·플로럴·그래픽·애니멀·카모·컬러블록 (9) | 0.45 |
| `material` | **다중** | 코튼·데님·니트·울·가죽·스웨이드·시폰·실크새틴·퍼플리스·메시·레이스 (11) | 0.48 |
| `detail` | **다중** | 없음·포켓·프릴·지퍼·벨트·단추·리본·레이스·플리츠·자수·스팽글·후드 (12) | 0.45 |

**하의 4축 분해 설계**가 이 스키마의 핵심 아이디어입니다:
`팬츠` 하나로 뭉치지 않고 → `종류(카고) × 다리모양(와이드) × 기장(풀렝스) × 디테일(카고포켓)` 로 조합.
상의는 `UPPER_ONLY_TASKS`, 하의는 `LOWER_ONLY_TASKS`로 나뉘어 카테고리에 맞지 않는 태스크는 애초에 라벨링/추론하지 않습니다.

### 2-6. 학습 하이퍼파라미터 (현재 체크포인트 실측)

`models/fashion_attribute_heads.metrics.json` 의 `config`:

```json
{ "epochs": 50, "batch_size": 128, "learning_rate": 0.0005,
  "weight_decay": 0.0001, "hidden_dim": 256, "dropout": 0.3,
  "patience": 8, "seed": 42, "minimum_label_examples": 5 }
```

- optimizer: AdamW, LR 스케줄 없음(고정 LR)
- **태스크별 개별 early stopping**: `best_task_losses`에 태스크별 최저 val loss 시점의 헤드 가중치를 따로 저장 → 마지막에 태스크별 최적 가중치를 조립
- 실제 26 epoch에서 종료 (patience 8)

---

## 3. 어떤 평가지표를 사용하는가

### 3-1. 학습 파이프라인 내부 지표 — `_metrics()` (fashion_attribute_training.py:366)

**단일분류 태스크 (13개)**

| 지표 | 정의 |
|---|---|
| `accuracy` | 전체 val 표본에 대한 argmax 정확도 (임계값 무시) |
| `accepted_coverage` | confidence ≥ threshold 인 표본 비율 = **"응답률"** |
| `accepted_accuracy` | 그중에서의 정확도 = **"응답한 것만의 정확도"** |
| `per_label.recall` | TP / support |
| `per_label.precision` | TP / 예측 수 |

**다중분류 태스크 (4개)**

| 지표 | 정의 |
|---|---|
| `micro_f1` | 전 라벨 TP/FP/FN 합산 → `2TP / (2TP+FP+FN)` |
| `per_label.f1` | 라벨별 F1 |
| `threshold` | 검증셋에서 F1 최대가 되도록 선택된 값 |

> ⚠️ macro-F1 은 여기서 계산하지 않습니다(라벨별 값만 저장). 그래서 **소수 클래스 실패가 대표 숫자에 잘 안 드러납니다.**

### 3-2. 임계값 튜닝 방식 (2종)

1. `_tune_multilabel_thresholds()` — 다중분류 4태스크. 0.30~0.70을 0.05 간격으로 훑어 **micro-F1 최대** 지점 선택
2. `_tune_pants_single_thresholds()` — `lower_subtype`, `pant_leg_shape`, `pant_length` 3태스크만 특별 취급.
   0.45~0.85 중 **응답률 ≥ 20%** 를 만족하면서 **정확도 ≥ 0.80** 이 되는 가장 낮은 임계값 선택
   → "틀린 답을 내놓기보다 보류한다"는 정책

> ⚠️ **주의: 임계값이 val 셋에서 선택되고 성능도 val 셋에서 보고됩니다.** 즉 현재 README의 숫자는 낙관적으로 편향돼 있습니다. (1단계에서 이 문제를 다룹니다)

### 3-3. 별도 외부 평가 경로 — DeepFashion

- `deepfashion_dataset.py` + `deepfashion_evaluation.ipynb`
- **DeepFashion-MultiModal**(비상업 연구용, 레포에 미포함 / 별도 동의 필요)로
  `소매 길이 / 하의 길이 / 패턴 / 소재 / 네크라인` 5개만 교차 평가
- 지표: **응답률, 선택 정확도, 전체 정확도, macro-F1, 대표 오답**
- 결과: `outputs/deepfashion_evaluation.json`
- 현재 이미지가 없어 **실행 이력 없음**

### 3-4. 테스트 코드

`tests/` 8개 파일 — pytest. 성능 지표가 아니라 **경로/스키마/규칙 로딩/추천 로직의 계약(contract) 검증**용입니다.
(`test_fashion_attribute_heads.py`, `test_fashion_rules.py`, `test_recommendation.py` 등)

---

## 4. 현재 체크포인트 성능 (val 1,195 crop, 기준선)

정렬: 점수 낮은 순 = **약한 순**

| # | task | 지표 | **점수** | 응답률 | 응답분 정확도 | val 표본 |
|---|---|---|---|---|---|---|
| 1 | `pant_leg_shape` | acc | **0.570** | 32.1% | 0.850 | 374 |
| 2 | `neckline` | acc | **0.635** | 80.0% | 0.706 | 85 |
| 3 | `upper_fit` | acc | **0.640** | 88.2% | 0.650 | 178 |
| 4 | `lower_fit` | acc | **0.670** | 80.5% | 0.725 | 442 |
| 5 | `silhouette` | acc | **0.703** | 76.2% | 0.818 | 101 |
| 6 | `material` | micro-F1 | **0.706** | – | – | 156 |
| 7 | `lower_subtype` | acc | **0.706** | 72.6% | 0.803 | 252 |
| 8 | `detail` | micro-F1 | **0.720** | – | – | 262 |
| 9 | `collar` | acc | **0.736** | 87.4% | 0.803 | 87 |
| 10 | `pant_length` | acc | **0.743** | 72.1% | 0.819 | 222 |
| 11 | `upper_length` | acc | **0.742** | 99.2% | 0.748 | 132 |
| 12 | `sleeve_shape` | acc | **0.795** | 90.2% | 0.827 | 122 |
| 13 | `lower_length` | acc | **0.825** | 94.6% | 0.850 | 422 |
| 14 | `category` | acc | **0.870** | 96.8% | 0.886 | 1,195 |
| 15 | `lower_detail` | micro-F1 | **0.881** | – | – | 110 |
| 16 | `pattern` | micro-F1 | **0.884** | – | – | 389 |
| 17 | `sleeve_length` | acc | **0.893** | 96.7% | 0.903 | 214 |

### 눈에 띄는 이상 신호 (1단계에서 파고들 지점)

- **명확한 과적합**: train_loss 1.129 → 0.049 인데 val_loss는 epoch 3에서 0.607로 최저를 찍고 다시 0.853까지 상승
- **`7부 소매` recall 0.071** — sleeve_length 전체는 0.893으로 좋아 보이지만 이 라벨만 사실상 붕괴 (14개 중 1개만 맞춤)
- **`upper_length 기본 기장` recall 0.333** — 크롭(support 91)에 흡수됨
- **`팔라초` recall 0.174** — 와이드와 구분 실패
- **micro-F1이 가리는 것**: `pattern` 0.884는 `무지`(support 304) 덕분. `그래픽` F1 0.0, `애니멀` F1 0.0
- **`메시` F1 0.0, `스팽글` F1 0.0, `밴딩 허리` F1 0.0**
- **`오버핏`, `O라인`, `피터팬 칼라`, `래글런 소매` recall 0.0**

---

## 5. 파일 지도 (한 줄 설명)

### 학습·모델 (★ = 이번 작업 핵심)

| 파일 | 역할 |
|---|---|
| ★ `fashion_attribute_schema.py` | 17 태스크 × 124 라벨 정의, 단일/다중 여부, 최소 신뢰도 |
| ★ `fashion_attribute_dataset.py` | CSV 로딩, Fashionpedia JSON → CSV 변환, 영문→한국어 라벨 매핑 |
| ★ `fashion_attribute_model.py` | 헤드 구조, 체크포인트 저장/로딩, 추론 게이팅, 측정값 융합 |
| ★ `fashion_attribute_training.py` | 임베딩 캐시, 손실, 클래스 가중치, 학습 루프, 지표, 임계값 튜닝 |
| `train_fashion_attribute_heads.py` | CLI (`convert-fashionpedia` / `cache` / `merge-cache` / `filter-cache` / `train`) |
| `train_fashion_attribute_heads.ipynb` | 위 CLI의 노트북 버전 |
| `prepare_fashionpedia_seed.py` | Fashionpedia 주석·이미지 추출 |
| `prepare_fashion200k_supplement.py` | Fashion200K 상의·소재 보완 |
| `prepare_fashion200k_bottoms.py` | Fashion200K 하의 4축 보완 |

### 추론·응용

| 파일 | 역할 |
|---|---|
| `main.ipynb` | 전체 실행 오케스트레이션 (경로설정 → 품질검사 → 체형 → 착장분석 → 추천 → 출력) |
| `fashion_model.py` | FashionSigLIP 로딩 + 학습 헤드 추론 (+ zero-shot fallback) |
| `fashion_prompts.py` | zero-shot용 텍스트 프롬프트 후보 |
| `clothing_parser.py` | FASHN Human Parser 마스크 |
| `garment_attribute_analyzer.py` | 마스크 × 관절 → 기하학적 길이/핏 측정 |
| `pose_analyzer.py` | MediaPipe 체형 비율 |
| `outfit_analyzer.py` | 색상 + 속성 통합 |
| `recommendation_engine.py` | 규칙 기반 필터·점수·설명 |
| `fashion_rules.py` | `FASHION_RULES_MASTER.md` R-* 규칙 파싱 |
| `product_catalog.py` / `data/products.csv` | 샘플 상품 (실제 쇼핑몰 아님) |
| `virtual_tryon.py` | VTON 인터페이스 (현재 비합성 추천 보드) |
| `quality_checker.py` / `feedback_store.py` | 입력 품질 검사 / 피드백 JSONL |

### 평가

| 파일 | 역할 |
|---|---|
| `deepfashion_dataset.py` | DeepFashion-MultiModal 라벨 로딩 + 교차 평가 |
| `deepfashion_evaluation.ipynb` | 평가 실행 노트북 |
| `models/*.metrics.json` | 학습 시 저장된 지표 리포트 |

---

## 6. 한계 (README + 코드에서 확인된 것)

1. **약지도 라벨**: Fashion200K 보강분 3,952행(66%)은 사람이 사진을 검수한 정답이 아니라 **상품명/카테고리 문자열 파싱** 결과
2. **val 이중 사용**: 임계값 선택과 early stopping, 성능 보고가 모두 같은 val 셋 → 보고 숫자가 낙관적
3. **도메인 갭**: 학습 데이터는 쇼핑몰 상품컷. 실제 사용자 **한국인 전신사진**에 대한 검증 없음
4. **본질적 모호성**: 슬랙스 vs 치노, 스트레이트 vs 테이퍼드 — 원단·주름이 가려지면 사진만으로 구분 불가
5. **백본 freeze**: FashionSigLIP 자체는 미세조정하지 않음 → 임베딩이 구분하지 못하는 속성은 헤드가 아무리 커도 한계
6. **환경**: 현재 PC 기본 파이썬(3.13)에 `torch` 미설치 (임베딩 캐시는 이미 있어 헤드 재학습만은 가벼움)

---

## 7. 다음 단계로 넘길 결론

- 학습 대상은 **캐시된 768차원 임베딩 위의 17개 MLP 헤드** 하나뿐 → **재학습이 매우 저렴**(CPU 수 분)
- 개선 여지가 가장 큰 곳: `pant_leg_shape`(0.570), `neckline`(0.635), `upper_fit`(0.640), `lower_fit`(0.670)
- 대표 숫자에 가려진 붕괴 라벨: `7부 소매`, `기본 기장`, `팔라초`, `그래픽`, `애니멀`, `메시`, `스팽글`, `오버핏`
- 명확한 **과적합**(val_loss 0.607 → 0.853)과 **val 이중 사용** 문제 → 1단계에서 정직한 held-out 분할부터 만들어야 함
