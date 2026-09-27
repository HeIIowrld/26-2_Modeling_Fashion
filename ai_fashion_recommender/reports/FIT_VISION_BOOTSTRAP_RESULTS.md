# 하의 핏 4종 GPU 파인튜닝 결과

## 실험 범위

`슬림 / 스트레이트 / 세미와이드 / 와이드` 네 클래스를 대상으로 FashionSigLIP 백본을 고정하고 분류 헤드만 학습했다. 스키니와 테이퍼드는 사용자가 정한 기준에 따라 슬림으로 통합했다. 세부 판정 기준은 [FIT_VISION_BOOTSTRAP_CRITERIA.md](FIT_VISION_BOOTSTRAP_CRITERIA.md)에 정리했다.

| 항목 | 값 |
|---|---:|
| 전체 표본 | 800 |
| train / validation / test | 560 / 120 / 120 |
| 클래스별 표본 | 200 |
| 사용자형 / 쇼핑형 | 403 / 397 |
| 학습 장치 | NVIDIA GPU, CUDA |
| 백본 | `Marqo/marqo-fashionSigLIP` 고정 |
| 검수 상태 | 자동 약지도, 사람 검수 대기 |

동일 의상 또는 상품은 `group_id` 단위로 한 split에만 들어간다. DeepFashion-MultiModal은 공식 바지 마스크를 사용했고, Fashion200K는 FASHN human parser로 착용 사진을 선별해 마스크를 만들었다.

## Test 결과

표의 macro-F1은 전체 7개 런타임 라벨 중 이번 실험에 실제로 포함된 네 클래스만 평균한 `active_macro_f1`이다.

| 구성 | 정확도 | 4종 macro-F1 | 사용자 macro-F1 | 쇼핑 macro-F1 |
|---|---:|---:|---:|---:|
| `rgb_only` | **0.8083** | **0.8066** | 0.9344 | **0.6737** |
| `masked_only` | 0.7833 | 0.7811 | 0.9514 | 0.5991 |
| `rgb_mask` | 0.7917 | 0.7906 | **0.9516** | 0.6246 |
| `rgb_geometry` | **0.8083** | **0.8066** | 0.9344 | **0.6737** |
| `rgb_mask_geometry` | 0.7917 | 0.7906 | **0.9516** | 0.6246 |

| 구성 | 슬림 F1 | 스트레이트 F1 | 세미와이드 F1 | 와이드 F1 |
|---|---:|---:|---:|---:|
| `rgb_only` | 0.8923 | **0.7333** | **0.7586** | **0.8421** |
| `masked_only` | **0.9063** | 0.7018 | 0.7385 | 0.7778 |
| `rgb_mask` | 0.8750 | 0.7119 | 0.7541 | 0.8214 |
| `rgb_geometry` | 0.8923 | **0.7333** | **0.7586** | **0.8421** |
| `rgb_mask_geometry` | 0.8750 | 0.7119 | 0.7541 | 0.8214 |

## 판정

`rgb_mask_geometry`는 RGB 기준선보다 test macro-F1이 0.0160 낮았다. 마스크와 기하 정보를 사용하는 후보를 서비스 모델로 승격하지 않는다. 현재 연구용 최선은 더 단순한 `rgb_only` 체크포인트다.

사용자형 test 점수가 쇼핑형보다 크게 높다. 자동 선별에 사용한 FashionSigLIP 및 마스크 기하와 학습 입력이 연관돼 있으므로, 이 수치는 실제 사람 라벨 일반화 성능으로 해석할 수 없다. 다음 단계는 경계 표본, 특히 쇼핑 스트레이트/세미와이드를 사람이 검수한 뒤 독립 test set으로 다시 평가하는 것이다.

## 로컬 산출물

원본 이미지, 마스크, 임베딩 캐시, 체크포인트는 라이선스와 용량 때문에 Git에 커밋하지 않는다.

| 산출물 | 로컬 상대 위치 (`work/fit-data`) |
|---|---|
| 주석과 manifest | `dataset/annotations.csv`, `dataset/manifest.json` |
| 검수 시트 | `dataset/review/` |
| 임베딩 캐시 | `cache/train.pt`, `cache/val.pt`, `cache/test.pt` |
| 5개 체크포인트와 지표 | `outputs/fit_<feature_mode>.pt`, `outputs/fit_<feature_mode>.metrics.json` |

모든 체크포인트 메타데이터에는 `research_only_until_human_label_review`가 기록되어 있다.
