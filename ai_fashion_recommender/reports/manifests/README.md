# 데이터 출처 매니페스트

학습·평가에 쓰인 **모든 원본 이미지**를 이미지 단위(crop 아님)로 기록합니다.
절대경로는 저장하지 않으며, 이 폴더 전체는 GitHub에 올릴 수 있습니다.

## 파일

| 파일 | 행 | 내용 |
|---|---|---|
| `used_image_manifest.csv` | 15,610 | **통합 매니페스트.** 이미지당 1행 |
| `fashionpedia_train_r1_images.csv` | 4,500 | 1차 보강 선택분 전체 |
| `fashionpedia_train_r2_images.csv` | 6,000 | 2차 보강 선택분 전체 |
| `fashionpedia_val_seed_images.csv` | 1,158 | 기존 seed (val2020) |
| `fashion200k_images.csv` | 3,952 | Fashion200K 두 서브셋 |
| `fashionpedia_train_excluded_image_ids.json` | 10,500 | **향후 후보에서 제외할 image_id** |
| `sources.json` | — | 원본 URL·SHA256 |

## 고유키 규칙

```
Fashionpedia            fashionpedia::{official_split}::{image_id}
Fashionpedia (id 미복원) fashionpedia::{official_split}::{original_filename}
Fashion200K             fashion200k::{source_subset}::{product_id}::{image_number}
```

현재 **미복원 항목은 0건**이므로 모든 Fashionpedia 행이 image_id 기반 키를 씁니다.

## usage_split

| 값 | 이미지 | crop | 의미 |
|---|---|---|---|
| `train` | 13,680 | 22,341 | 학습에 사용 |
| `old_val` | 1,020 | 1,195 | 기존 validation set |
| `new_val` | 899 | 1,716 | 새 validation set |
| `selected_no_valid_crop` | 11 | 0 | 선택됐으나 라벨 가능한 의류가 없어 crop 0개 |

`used_for_training=true` 는 `usage_split=train` 인 행뿐입니다.

## 향후 3차 보강 시 주의

**제외 대상은 "학습에 쓴 이미지"가 아니라 "선택된 image_id 전체"입니다.**
crop이 0이었던 11개도 이미 소진된 후보이므로 `fashionpedia_train_excluded_image_ids.json`
(10,500개)을 통째로 제외해야 합니다. train2020 전체 45,623개 중 **남은 후보는 35,123개**입니다.

validation 이미지(`old_val`, `new_val`)는 **어떤 경우에도 학습 후보에 넣으면 안 됩니다.**

## 원본 보존 위치 (gitignored, 로컬 전용)

```
data/provenance/fashionpedia/r1/instances_subset.json    image_id ↔ 파일명 (1차)
data/provenance/fashionpedia/r2/instances_subset.json    image_id ↔ 파일명 (2차)
data/provenance/fashionpedia/seed/instances_attributes_val2020.json
data/provenance/fashionpedia/shard_index.json            image_id → parquet shard (0.5MB, 커밋 가능)
```

`instances_attributes_train2020.json`(542MB)은 보존하지 않습니다.
r1/r2 `instances_subset.json`이 그 파생본이라 필요한 매핑은 모두 들어 있습니다.

## 재생성

```bash
cd experiments
python build_provenance_manifests.py
python verify_provenance_manifests.py
```

## 알려진 정정

`data/fashionpedia_train/manifest_r2.json`은 `shards_used`와 `selection` 값이 **틀렸습니다**
(생성 스크립트가 1차 값을 하드코딩). 원본은 감사 목적으로 그대로 두고,
**`manifest_r2_corrected.json`을 참조하세요.** 생성 스크립트(`build_train_csv.py`)는
shard 목록을 인자로 받도록 수정해 재발을 막았습니다.
