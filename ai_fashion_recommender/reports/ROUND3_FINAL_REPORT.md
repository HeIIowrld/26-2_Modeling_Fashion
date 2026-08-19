# 3차 보강 결과 리포트

> 실행일: 2026-08-19 · CPU 전용 (Python 3.13.0, torch 2.13.0+cpu)
> **결론: 3차 모델을 최종 모델로 채택하지 않습니다.** 2차 모델(`fashion_attribute_heads_augmented.pt`)을 유지합니다.

---

## 0. 채택 여부 — 미채택

| 채택 기준 | 결과 | 판정 |
|---|---|---|
| old validation mean score | 0.7641 → **0.7593** (−0.0048) | ❌ 하락 |
| old validation mean macro-F1 | 0.6726 → **0.6641** (−0.0085) | ❌ 하락 |
| 3개 seed 안정성 | mean 0.7610 ± 0.0043 | ✅ 안정 |
| 주요 희소 라벨 붕괴 | 붕괴 없음 (오버핏 0.235→0.308, 후드티 0.667→1.000) | ✅ |
| coverage 감소 | `pant_leg_shape` −0.556 (원인은 임계값, 5-4 참조) | ⚠️ |

**old validation 두 대표 지표가 모두 하락**했으므로 지시대로 억지 채택하지 않습니다.
new validation에서는 올랐지만(+0.0074 / +0.0105) 그 셋은 희소 라벨 편향 선별본이라
3차 보강과 분포가 겹칩니다. 자기 유리한 셋의 상승을 근거로 채택할 수 없습니다.

---

## 1. 선택

| 항목 | 값 |
|---|---|
| 후보 풀 | 45,623 − 제외 10,500 = **35,123** |
| 선택 | **6,000장** (상한 도달, 조기 중단 아님) |
| shard 분포 | 0:540 / 1:548 / 2:1,067 / 3:952 / 4:964 / 5:967 / 6:962 (전 shard 사용) |
| random seed | 20260819 |
| 점수 대상 라벨 | 84개 (Fashionpedia 미존재 라벨·하의 4축 제외) |

우선순위 가중치: `<30` 100 · `<100` 40 · `<300` 12 · `<800` 4 · `<1500` 1 · 그 이상 0.
같은 단계에서는 `1 + (1 − F1)`을 곱해 기존 모델의 F1이 낮은 라벨을 더 크게 평가했습니다.
한 이미지가 여러 부족 라벨을 채우면 점수가 합산됩니다.

**1·2차와 달리 T1(<30)·T2(<100) 결핍 라벨은 이미 거의 소진**되어(각 1개),
이번 보강은 대부분 T4~T5(300~1,500) 구간을 채웠습니다.

주요 라벨 보강 (전 → 후):

| 라벨 | 전 | 후 |
|---|---|---|
| `lower_fit|플레어핏` | 476 | 1,500 |
| `silhouette|X라인` | 862 | 1,500 |
| `detail|리본` · `detail|플리츠` | 893 | 1,500 |
| `neckline|V넥` | 907 | 1,500 |
| `collar|폴로 칼라` | 130 | 251 |
| `collar|피터팬 칼라` | 139 | 193 |
| `silhouette|O라인` | 104 | 160 |
| `pattern|카모` | 109 | 139 |

교집합 검사 (선택 직후): r3∩r1 **0** · r3∩r2 **0** · r3∩old val **0** · r3∩new val **0** · r3 내부 중복 **0**

## 2. 추출·CSV·임베딩

| 항목 | 값 |
|---|---|
| 추출 | 6,000장 / 344초 / 누락 **0** |
| 잘못된 bbox | **0** (10,030 crop 전수 검사) |
| crop | **10,030** (train 10,030, crop 0인 이미지 2장) |
| 임베딩 | 72.7분, 2.30 crop/s |
| crop 수 == embedding 수 | ✅ 10,030 / 10,030 |
| NaN / Inf | ✅ 0 / 0 |
| L2 정규화 | ✅ norm 최소·최대 1.000000 |
| 차원 | (10030, 768) |

`build_train_csv.py`의 shard 하드코딩을 고친 뒤 실행해, 이번 manifest는
`shards_used=[0,1,2,3,4,5,6]`, `selection_target=1500`으로 **정확히 기록**되었습니다.

## 3. image-level fit/dev 분할

기존 `split_train_dev()`는 crop 단위라 같은 이미지가 fit/dev 양쪽에 들어갔습니다.
`experiments/image_level_split.py`가 canonical key 단위로 나눕니다.

```
Fashionpedia : fashionpedia::{official_split}::{image_id}
Fashion200K  : fashion200k::{source_subset}::{product_id}::{image_number}
```

| 검사 | 결과 |
|---|---|
| 캐시 crop 중 manifest 매핑 실패 | **0** |
| 하나의 crop이 복수 이미지에 매핑 | **0** |
| fit image key ∩ dev image key | **0** |
| fit/dev crop 합계 == 전체 train crop | 27,463 + 4,908 = **32,371** |
| fit ∩ old/new validation | **0 / 0** |
| dev ∩ old/new validation | **0 / 0** |

**같은 데이터를 기존 crop 단위로 나누면 dev crop의 61.3%(이미지 2,767장)가 fit과 겹칩니다.
image-level 분할은 0%입니다.**

## 4. 재학습

R2_mixup 그대로(hidden_dim 256, dropout 0.4, wd 1e-3, lr 5e-4 cosine, mixup α 0.4, batch 256, 100 epoch).
early stopping은 **새 image-level dev만** 사용, old/new validation은 학습·선택 어디에도 쓰지 않았습니다.

| seed | dev composite | 소요 |
|---|---|---|
| 0 | 0.7583 | 1,908s |
| **1 (선택)** | **0.7588** | 2,130s |
| 2 | 0.7580 | 1,870s |

최종 체크포인트는 **dev composite 최고인 seed 1**로 선택했습니다(validation 점수 미사용).

## 5. 평가

### 5-1. old validation 1,195 crop — 하락

| 모델 | mean score | 표본가중 | mean macro-F1 |
|---|---|---|---|
| 2차 (현행) | **0.7641** | **0.7826** | **0.6726** |
| 3차 seed0 | 0.7669 | 0.7877 | 0.6844 |
| 3차 seed1 (최종) | 0.7593 | 0.7796 | 0.6641 |
| 3차 seed2 | 0.7567 | 0.7831 | 0.6655 |
| **3 seed 평균±표준편차** | 0.7610 ± 0.0043 | — | 0.6713 ± 0.0093 |
| **최종 3차 − 2차** | **−0.0048** | −0.0030 | **−0.0085** |

> ⚠️ **주목할 점**: seed 0은 2차보다 **좋았습니다**(0.7669 / 0.6844). 그런데 dev composite는
> seed 1이 0.0005 높았고(0.7588 vs 0.7583) 그 seed가 old validation에서는 더 나빴습니다.
> **dev와 old validation의 순위가 어긋납니다.** dev로 고르는 원칙은 유지했지만,
> dev composite 차이 0.0005는 선택 근거로 삼기에 너무 작습니다.

태스크별 (2차 → 3차 최종): 상승 5 / 동일 2 / 하락 10

| 상승 | Δ | 하락 | Δ |
|---|---|---|---|
| `neckline` | +0.035 | `sleeve_shape` | −0.041 |
| `upper_fit` | +0.034 | `silhouette` | −0.040 |
| `lower_length` | +0.009 | `pant_leg_shape` * | −0.024 |
| `detail` | +0.006 | `material` | −0.016 |
| `category` | +0.002 | `pant_length` * | −0.013 |

`*` = Fashionpedia에 라벨이 없어 **3차에서 새 표본을 한 건도 받지 못한** 태스크
(`lower_subtype` `pant_leg_shape` `pant_length` `lower_detail`).
**이 네 축의 변화는 데이터 보강 효과가 아니라 학습 변동입니다.**

### 5-2. new validation 1,716 crop — 상승

| 모델 | mean score | 표본가중 | mean macro-F1 |
|---|---|---|---|
| 2차 | 0.7651 | 0.7717 | 0.6785 |
| 3차 seed0 | 0.7677 | 0.7733 | 0.6927 |
| 3차 seed1 (최종) | **0.7725** | **0.7760** | **0.6890** |
| 3차 seed2 | 0.7677 | 0.7734 | 0.6884 |
| **3 seed 평균±표준편차** | 0.7693 ± 0.0023 | — | 0.6900 ± 0.0019 |
| **최종 3차 − 2차** | **+0.0074** | +0.0043 | **+0.0105** |

13개 태스크 중 9개 상승. `silhouette` +0.034, `sleeve_shape` +0.026, `lower_fit` +0.019.

> 이 셋은 희소 라벨이 많이 들어가도록 선별된 데이터라 **실사용 분포를 대표하지 않으며**,
> 3차 보강분과 같은 계열(Fashionpedia train2020 greedy 선별)입니다.
> 여기서의 상승은 **보강 분포에 더 잘 맞게 된 결과**일 수 있어 채택 근거로 쓰지 않았습니다.

### 5-3. 주요 라벨 precision / recall / F1

new validation 기준 (표본이 충분한 쪽):

| 라벨 | n | 2차 | 3차 |
|---|---|---|---|
| `category|니트` | 54 | 0.621 / 0.667 / **0.643** | 0.586 / 0.630 / 0.607 ▼ |
| `category|가디건` | 40 | 0.691 / 0.725 / **0.707** | 0.667 / 0.700 / 0.683 ▼ |
| `category|후드티` | 19 | 0.632 / 0.632 / 0.632 | 0.684 / 0.684 / **0.684** ▲ |
| `detail|단추` | 202 | 0.815 / 0.827 / 0.821 | 0.806 / 0.861 / **0.833** ▲ |
| `sleeve_shape|퍼프 소매` | 30 | 0.464 / 0.433 / 0.448 | 0.609 / 0.467 / **0.528** ▲ |
| `upper_fit|오버핏` | 50 | 0.318 / 0.540 / 0.400 | 0.429 / 0.420 / **0.424** ▲ |
| `collar|피터팬 칼라` | 19 | 0.467 / 0.368 / 0.412 | 0.444 / 0.421 / **0.432** ▲ |

**희소 라벨 붕괴는 없습니다.** 다만 니트·가디건이 소폭 하락했습니다.

### 5-4. coverage 급락의 원인 — 모델 붕괴가 아닙니다

old validation `pant_leg_shape` coverage 0.941 → **0.385**(−0.556)이지만
같은 구간의 **accepted accuracy는 0.631 → 0.785로 올랐습니다.**

원인은 하의 3축 임계값 자동 튜너입니다. "정확도 0.80을 만족하는 최저 임계값"을 고르는데,
seed마다 다른 값을 골랐습니다.

| 모델 | `pant_leg_shape` 임계값 |
|---|---|
| 2차 | 0.45 |
| 3차 seed0 | 0.80 |
| 3차 seed1 (최종) | 0.65 |
| 3차 seed2 | 0.45 |

`pant_leg_shape`는 **3차에서 새 표본을 받지 못한 태스크**라 seed 간 변동이 그대로 임계값에
반영된 것입니다. 모델 성능 붕괴가 아니라 **coverage-accuracy 트레이드오프가 seed마다 다르게
찍힌 결과**이며, 이 불안정성 자체가 3차 미채택의 보조 근거입니다.

## 6. 산출물

| 파일 | 내용 |
|---|---|
| `reports/16_round3_selection.json` | 선택 근거·라벨 전후 |
| `reports/17_round3_split_integrity.json` | 분할 검사·seed 결과·SHA256 |
| `reports/17_round3_extraction_issues.json` | 누락 0 / 불량 bbox 0 |
| `reports/18_round3_comparison_oldval.json` | old validation 비교 |
| `reports/19_round3_comparison_newval.json` | new validation 비교 |
| `reports/manifests/fashionpedia_train_r3_images.csv` | 6,000행 |

**SHA256**

| 모델 | SHA256 |
|---|---|
| 2차 (학습 전·후 동일) | `3c560c9db0ade6a62d3ead1fc61edc56ff6a935083ee7393858a3a757431765d` |
| 3차 최종 (seed1) | `399c65242147ffcdbe83d5aef09edf05…` |
| 3차 seed0 | `076aa2c49cd64fd9ba54d92e63a601a6…` |
| 3차 seed1 | `a821cd8849c64c3e33b879466f031ce7…` |
| 3차 seed2 | `78b66b0b15cbfb5715f1ae6667ec4d02…` |

**실행 시간**: 추출 344초 + 임베딩 72.7분 + 학습 98.6분 ≈ **3시간**

## 7. 남은 한계와 해석

1. **3차 보강은 old validation을 개선하지 못했습니다.** 1→2차에서 macro-F1이 +1.5pt 올랐던 것과 달리
   2→3차는 −0.85pt입니다.

   다만 해석은 다음 범위로 제한합니다 —
   **현재의 Fashionpedia 희소 라벨 중심 선택 방식과 R2_mixup 학습 설정에서는 추가 보강의 일관된
   성능 향상이 확인되지 않았습니다. seed 0은 개선됐지만 3개 seed 평균과 dev 기준 선택 모델은
   2차 모델을 안정적으로 넘지 못했으므로, Fashionpedia 데이터 자체의 한계효용이 완전히
   소진됐다고 단정할 수는 없습니다.**

   T1·T2 결핍 라벨이 이미 소진되어 이번 보강 대부분이 이미 800~1,500개인 라벨에 들어간 것이
   한 요인으로 보이지만, 선택 방식·학습 설정·dev 구성 중 무엇이 지배적인지는 이번 실험만으로
   분리되지 않습니다.
2. **dev와 old validation의 순위가 어긋납니다.** dev composite 1위 seed가 old validation에서는 3위입니다.
   image-level dev로 누수는 없앴지만, dev(Fashionpedia 비중이 큼)와 old validation(Fashion200K 절반)의
   **분포 차이**가 남아 있습니다.
3. **하의 세부 4축은 3차에서도 새 표본을 받지 못했습니다.** 이 축들의 등락은 전부 학습 변동입니다.
4. **new validation은 채택 판단에 쓸 수 없습니다.** 선별 편향 때문에 3차 보강에 유리합니다.
5. 임계값 자동 튜너가 seed에 민감합니다(`pant_leg_shape` 0.45~0.80). 튜닝 안정화가 필요합니다.

**다음에 시도할 만한 것** — Fashionpedia 4차보다 우선순위가 높습니다.

- Fashion200K 하의 보강 (유일하게 데이터를 못 받은 영역)
- 약지도 라벨 재검수로 달성 가능한 상한선 확정
- dev를 old validation과 같은 출처 비율로 구성해 선택 신뢰도 높이기
- 임계값 튜닝을 seed 평균으로 안정화

---

## 8. 실행 상태 (배포·재현 기준)

### 8-1. 현재 채택 모델

| 항목 | 값 |
|---|---|
| **배포 모델** | `models/fashion_attribute_heads_augmented.pt` (**2차**) |
| 배포 모델 학습 crop | **22,341** (기존 4,789 + 1차 6,797 + 2차 10,755) |
| 3차 모델 | 실험 완료, **미채택** |
| 3차 실험용 train crop | **32,371** (위 22,341 + 3차 10,030) |
| main 코드 모델 경로 | **변경하지 않음** |

> ⚠️ **혼동 주의: 배포 모델은 3차 데이터를 학습하지 않았습니다.**
> `fashion_attribute_heads_augmented.pt`는 **22,341 crop**으로 학습된 2차 모델입니다.
> 32,371 crop은 3차 실험(`fashion_attribute_heads_augmented_r3*.pt`)에만 해당하며,
> 이 체크포인트들은 저장소에 포함되지 않습니다.

### 8-2. 3차 미채택 근거 정리

- **seed 0은 old validation에서 2차 모델보다 높았습니다** (mean 0.7669 vs 0.7641, macro-F1 0.6844 vs 0.6726).
- 그러나 **dev 기준 seed 선택 원칙을 사후에 바꾸지 않았습니다.** dev composite 1위는 seed 1이었고
  (0.7588 vs 0.7583), 결과가 나온 뒤 old validation 점수를 보고 seed 0으로 갈아타는 것은
  validation을 모델 선택에 쓰는 것이므로 하지 않았습니다.
- **3개 seed 평균은 2차를 안정적으로 넘지 못했습니다** (old validation mean 0.7610±0.0043 vs 0.7641,
  macro-F1 0.6713±0.0093 vs 0.6726).
- **new validation은 희소 라벨 편향 선별본**이라 3차 보강분과 분포가 겹칩니다. 채택 근거에서 제외했습니다.
- 따라서 **3차 모델은 실험 결과로만 보존하고 배포하지 않습니다.**

### 8-3. 3차 재현 순서

```bash
cd ai_fashion_recommender/experiments

# 0) 사전: 원본 주석·이미지 확보 (저장소에 없음, 8-5 참조)
python scan_fashionpedia_train.py <instances_attributes_train2020.json> <scan.json>

# 1) 선택  → reports/manifests/fashionpedia_train_r3_images.csv, reports/16_round3_selection.json
python select_round3.py

# 2) 추출  → data/fashionpedia_train_r3/images/
python extract_train_images.py     ../data/fashionpedia_train_r3/selection_r3.json <annot.json>     ../data/fashionpedia_train_r3 0,1,2,3,4,5,6

# 3) CSV   → data/fashionpedia_train_r3/fashion_attribute_annotations.csv
python build_train_csv.py     ../data/fashionpedia_train_r3/selection_r3.json <annot.json>     ../data/fashionpedia_train_r3 0.0 0,1,2,3,4,5,6

# 4) 임베딩 → data/cache/fashion_attribute_embeddings_r3.pt  (CPU 약 73분)
python embed_round3.py --threads 10

# 5) 학습   → models/fashion_attribute_heads_augmented_r3*.pt  (CPU 약 99분)
python train_round3.py --seeds 0 1 2 --threads 10

# 6) 평가   → reports/18_, 19_round3_comparison_*.json
python evaluate_round3.py
```

provenance 매니페스트를 다시 만들려면:

```bash
python build_provenance_manifests.py
python verify_provenance_manifests.py
```

### 8-4. 중복 방지 — manifest 사용법

4차 이후 후보를 고를 때는 **`reports/manifests/fashionpedia_train_excluded_image_ids.json`에
3차 선택분 6,000개를 합쳐** 제외해야 합니다. `select_round3.py`가 쓰는 방식과 동일합니다.

```python
import json, csv
excluded = set(json.load(open("reports/manifests/fashionpedia_train_excluded_image_ids.json"))["image_ids"])
excluded |= {int(r["image_id"]) for r in
             csv.DictReader(open("reports/manifests/fashionpedia_train_r3_images.csv", encoding="utf-8-sig"))}
# → 16,500개 제외, train2020 45,623 중 남은 후보 29,123
```

**제외 대상은 "학습에 쓴 이미지"가 아니라 "선택된 image_id 전체"입니다.**
crop이 0이었던 이미지(1차 4 · 2차 7 · 3차 2)도 이미 소진된 후보이므로 포함합니다.

`used_image_manifest.csv`의 `usage_split`이 `old_val` / `new_val`인 이미지는
**어떤 경우에도 학습 후보에 넣으면 안 됩니다.**

### 8-5. 원본 데이터 미포함 · 출처 표기

원본 이미지와 주석은 **용량과 라이선스 문제로 저장소에 포함하지 않습니다.**

| 항목 | 크기 | 저장소 포함 |
|---|---|---|
| Fashionpedia 이미지 (1·2·3차) | 약 1.2 GB | ❌ |
| `instances_attributes_train2020.json` | 542 MB | ❌ |
| `instances_attributes_val2020.json` | 14.5 MB | ❌ |
| parquet shard 7개 | 3.39 GB | ❌ (원격에서 직접 스트리밍) |
| 임베딩 캐시 | 약 100 MB | ❌ |
| **재현용 manifest (CSV/JSON)** | 약 3 MB | ✅ |

**Attribution**

> Fashionpedia — Menglin Jia et al., *Fashionpedia: Ontology, Segmentation, and an Attribute
> Localization Dataset* (ECCV 2020).
> 주석과 ontology는 **CC BY 4.0**입니다. 이미지 저작권은 원 저작자에게 있으며 재배포하지 않습니다.
> 주석 출처: `https://s3.amazonaws.com/ifashionist-dataset/annotations/`
> 이미지 parquet: `https://huggingface.co/datasets/detection-datasets/fashionpedia`
> 원본 SHA256은 `reports/manifests/sources.json`에 기록돼 있습니다.

> Fashion200K — Apache-2.0. `https://huggingface.co/datasets/Marqo/fashion200k`
> 상품 메타데이터에서 만든 **약지도 라벨**이며 사람이 사진을 재검수한 정답이 아닙니다.

> FashionSigLIP — `Marqo/marqo-fashionSigLIP`, Apache-2.0. 백본은 freeze 상태로만 사용했습니다.
