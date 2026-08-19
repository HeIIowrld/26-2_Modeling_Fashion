# data/ — 데이터 준비 안내

이 저장소에는 **라벨 CSV와 재현용 manifest만** 들어 있습니다.
원본 이미지·주석·임베딩 캐시는 용량과 라이선스 때문에 제외했습니다.

---

## 1. GitHub에서 제외된 데이터

| 대상 | 용량 | 이유 |
|---|---|---|
| `*/images/` 원본 이미지 11,363장 | 595.8 MB | 용량 + 이미지 저작권 (재배포 불가) |
| `cache/` 임베딩 캐시 16개 | 230.5 MB | 재생성 가능 |
| 미채택 모델 6개 | 82.0 MB | 실험 산출물 (배포 대상 아님) |
| `provenance/**/instances_subset.json` 4개 | 22.6 MB | 원본 주석 파생본 |
| `provenance/**/train2020·val2020 주석` | 32.2 MB | 원본 주석 (공식 배포처에서 직접 받을 것) |
| `input_person.jpg`, `outputs/*.jpg` | 0.26 MB | **개인 사진 및 그 파생 결과물** |

제외 총계 **11,497개 / 964.3 MB**. 저장소에 포함된 것은 **137개 / 41.3 MB**입니다.

> 실행 결과 이미지(`outputs/`)도 개인 사진에서 파생될 수 있어 저장소에 넣지 않습니다.
> 파이프라인 산출물이 어떤 형태인지는 [reports/16_real_photo_test.md](../reports/16_real_photo_test.md)의
> 텍스트 결과를 참고하세요.

## 2. 각 폴더의 역할

| 경로 | 저장소 포함 | 역할 |
|---|---|---|
| `fashion_attribute_annotations.csv` | ✅ | **통합 학습 라벨 5,984 crop** (17속성 × 124라벨) |
| `fashion_attribute_annotations_template.csv` | ✅ | 자체 라벨 작성용 빈 서식 |
| `fashionpedia_seed/` | csv·manifest만 | Fashionpedia **val2020** 기반 초기 데이터 2,032 crop |
| `fashion200k_supplement/` | csv·manifest만 | Fashion200K 상의·소재 보완 820 crop (약지도) |
| `fashion200k_bottoms/` | csv·manifest만 | Fashion200K 하의 4축 보완 3,132 crop (약지도) |
| `fashionpedia_train/` | csv·manifest만 | **1·2차 보강** — train2020에서 10,500장 선별 → 19,268 crop |
| `fashionpedia_train_r3/` | csv·manifest만 | **3차 보강(미채택)** — 6,000장 → 10,030 crop |
| `provenance/fashionpedia/` | `shard_index.json`만 | image_id ↔ 파일명 ↔ parquet shard 추적 |
| `cache/` | ❌ | FashionSigLIP 임베딩 (코드가 자동 생성) |
| `*/images/` | ❌ | 원본 이미지 (스크립트가 자동 생성) |
| `products.csv` | ✅ | 추천 로직 검증용 샘플 상품 (실제 쇼핑몰 아님) |
| `fashion_rules*.json` | ✅ | 패션 규칙 후보 |

## 3. 데이터 준비 방법

### 3-1. 초기 데이터 (Fashionpedia val2020 + Fashion200K)

```bash
cd ai_fashion_recommender
python prepare_fashionpedia_seed.py --parquet <val parquet> \
    --annotations <instances_attributes_val2020.json> --output-dir data/fashionpedia_seed
python prepare_fashion200k_supplement.py   # 인자는 스크립트 --help 참조
python prepare_fashion200k_bottoms.py
```

### 3-2. 1·2·3차 보강 (Fashionpedia train2020)

```bash
cd experiments
# 주석 스캔 (542MB JSON을 ijson 스트리밍으로 처리, 메모리 1GB 미만에서도 동작)
python scan_fashionpedia_train.py <instances_attributes_train2020.json> <scan.json>

# 선택 → 추출 → CSV  (3차 예시. 1·2차는 selection.json / selection_r2.json 사용)
python select_round3.py
python extract_train_images.py ../data/fashionpedia_train_r3/selection_r3.json \
    <annot.json> ../data/fashionpedia_train_r3 0,1,2,3,4,5,6
python build_train_csv.py ../data/fashionpedia_train_r3/selection_r3.json \
    <annot.json> ../data/fashionpedia_train_r3 0.0 0,1,2,3,4,5,6
```

> `build_train_csv.py`는 **shard 목록을 5번째 인자로 반드시** 받습니다.
> 예전에는 `[0,1]`로 하드코딩돼 2차 manifest에 잘못된 값이 기록됐습니다
> (교정본: `fashionpedia_train/manifest_r2_corrected.json`).

**중복 방지**: 새 후보를 고를 때는 이미 선택된 image_id를 반드시 제외하세요.

```python
import json, csv
excluded = set(json.load(open("reports/manifests/fashionpedia_train_excluded_image_ids.json"))["image_ids"])
excluded |= {int(r["image_id"]) for r in
             csv.DictReader(open("reports/manifests/fashionpedia_train_r3_images.csv", encoding="utf-8-sig"))}
# 1·2·3차 합계 16,500개 제외 → train2020 45,623 중 남은 후보 29,123
```

**제외 대상은 "학습에 쓴 이미지"가 아니라 "선택된 image_id 전체"입니다.**
crop이 0이었던 이미지(1차 4 · 2차 7 · 3차 2)도 이미 소진된 후보이므로 포함합니다.
`reports/manifests/used_image_manifest.csv`에서 `usage_split`이 `old_val` / `new_val`인
이미지는 **어떤 경우에도 학습 후보에 넣으면 안 됩니다.**

## 4. 캐시(임베딩) 생성 방법

FashionSigLIP은 freeze 상태로 이미지 임베딩만 만듭니다. CPU에서 약 **2.3 crop/s**입니다.

```bash
cd experiments
python embed_round3.py --threads 10          # 10,030 crop 기준 약 73분
```

- 출력: `data/cache/fashion_attribute_embeddings_r3.pt` (768차원 L2 정규화)
- 20배치마다 `*.progress.pt`에 중간 저장 → **중단해도 재개**됩니다
- 완료 시 crop 수 일치·NaN/Inf·L2 norm을 자동 검증합니다
- `data/cache/`는 코드가 자동 생성하므로 미리 만들 필요 없습니다

기존 1·2차 캐시는 `train_fashion_attribute_heads.py cache …` 로 만듭니다.

## 5. 원본 데이터 라이선스

> **Fashionpedia** — Menglin Jia et al., *Fashionpedia: Ontology, Segmentation, and an
> Attribute Localization Dataset* (ECCV 2020).
> 주석과 ontology는 **CC BY 4.0**. **이미지 저작권은 원 저작자에게 있으며 재배포하지 않습니다.**
> 주석: `https://s3.amazonaws.com/ifashionist-dataset/annotations/`
> 이미지 parquet: `https://huggingface.co/datasets/detection-datasets/fashionpedia`

> **Fashion200K** — Apache-2.0. `https://huggingface.co/datasets/Marqo/fashion200k`
> 상품 메타데이터에서 만든 **약지도 라벨**이며 사람이 사진을 재검수한 정답이 아닙니다.

> **FashionSigLIP** — `Marqo/marqo-fashionSigLIP`, Apache-2.0. 백본은 freeze로만 사용.

> **DeepFashion / DeepFashion-MultiModal** — 비상업 연구 전용, 재배포 제한.
> 저장소에 포함하지 않으며 평가 시 각자 사용 동의 절차를 거쳐야 합니다.

원본 파일의 SHA256은 [reports/manifests/sources.json](../reports/manifests/sources.json)에 있습니다.

## 6. 모델은 저장소에 포함됩니다

데이터와 달리 **최종 배포 모델은 저장소에 들어 있어 즉시 실행 가능합니다.**

| 파일 | 크기 | 용도 |
|---|---|---|
| `models/fashion_attribute_heads_augmented.pt` | 13.70 MB | **채택된 배포 모델** (2차 보강, train 22,341 crop) |
| `models/fashion_attribute_heads.pt` | 13.67 MB | 초기 baseline · rollback용 |

```python
# main.ipynb 경로 셀
ATTRIBUTE_HEADS_PATH_INPUT = r'models/fashion_attribute_heads_augmented.pt'
```

⚠️ **배포 모델은 3차 데이터를 학습하지 않았습니다.** 3차 실험(32,371 crop)은 미채택이며
체크포인트는 저장소에 없습니다. 근거는 [reports/ROUND3_FINAL_REPORT.md](../reports/ROUND3_FINAL_REPORT.md) 참조.

데이터 없이도 **모델 추론은 바로 됩니다.** 위 준비 과정은 **재학습할 때만** 필요합니다.
