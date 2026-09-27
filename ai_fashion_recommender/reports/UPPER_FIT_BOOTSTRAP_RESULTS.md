# 상의 핏 3종 GPU 학습 결과

## 실험 범위

사용자가 정한 `슬림핏 / 레귤러핏 / 오버핏` 세 클래스를 대상으로 FashionSigLIP 백본을 고정하고 분류 헤드만 학습했다. 기장은 핏과 분리했으며, 첫 학습에서는 레이어 경계가 불명확한 사진을 제외했다. 세부 판정 기준과 겉옷 정책은 [UPPER_FIT_BOOTSTRAP_CRITERIA.md](UPPER_FIT_BOOTSTRAP_CRITERIA.md)에 정리했다.

| 항목 | 값 |
|---|---:|
| 전체 표본 | 600 |
| train / validation / test | 420 / 90 / 90 |
| 클래스별 표본 | 200 |
| 사용자형 / 쇼핑형 | 300 / 300 |
| 학습 장치 | NVIDIA GeForce RTX 5060, CUDA |
| 백본 | `Marqo/marqo-fashionSigLIP` 고정 |
| 검수 상태 | 자동 선별, 사람 검수 대기 |

Fashionpedia는 명시적 핏 속성과 공식 인스턴스 마스크를 사용했다. `tight (fit)`은 슬림핏, `regular (fit)`은 레귤러핏, `loose (fit)`과 `oversized`는 오버핏으로 통합했다. DeepFashion-MultiModal에는 핏 정답이 없으므로 공식 상의 마스크와 FashionSigLIP 점수로 약한 라벨을 만들었다. 동일 인물·상품은 `group_id` 단위로 한 split에만 배치했다.

## Test 결과

표의 macro-F1은 런타임 상의 핏 4개 라벨 중 이번 실험에 포함된 세 클래스만 평균한 `active_macro_f1`이다. 사용하지 않은 `여유핏`은 0점으로 포함하지 않는다.

| 구성 | 정확도 | 3종 macro-F1 | 사용자 macro-F1 | 쇼핑 macro-F1 |
|---|---:|---:|---:|---:|
| `rgb_only` | 0.9889 | 0.9889 | 1.0000 | 0.9778 |
| `masked_only` | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| `rgb_mask` | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| `rgb_geometry` | 0.9889 | 0.9889 | 1.0000 | 0.9778 |
| `rgb_mask_geometry` | **1.0000** | **1.0000** | **1.0000** | **1.0000** |

| 구성 | 슬림핏 F1 | 레귤러핏 F1 | 오버핏 F1 |
|---|---:|---:|---:|
| `rgb_only` | 1.0000 | 0.9831 | 0.9836 |
| `masked_only` | **1.0000** | **1.0000** | **1.0000** |
| `rgb_mask` | **1.0000** | **1.0000** | **1.0000** |
| `rgb_geometry` | 1.0000 | 0.9831 | 0.9836 |
| `rgb_mask_geometry` | **1.0000** | **1.0000** | **1.0000** |

## 판정

`rgb_mask_geometry`는 RGB 기준선보다 test active macro-F1이 0.0111 높아 상의 단독 실험의 1차 채택 조건을 통과했다. 연구용 상의 후보 체크포인트는 `upper_fit_rgb_mask_geometry.pt`다.

이 점수는 실제 서비스 정확도로 해석할 수 없다. DeepFashion의 정답이 같은 FashionSigLIP 특징으로 생성됐고, Fashionpedia도 높은 점수의 쉬운 예시를 우선 선별했기 때문에 평가와 데이터 선택 신호가 연관돼 있다. 또한 test가 클래스·도메인별 15장으로 작고, 레이어드 착장은 제외돼 있다. 사람 라벨로 만든 독립 test set에서 재평가하기 전까지 체크포인트 상태는 연구용이다.

현재 공용 서비스 체크포인트는 교체하지 않는다. 기존 하의 실험에서는 RGB 기준선이 마스크·기하 구성보다 좋았으므로, 상의 전용 `rgb_mask_geometry` 파일로 공용 파일을 덮어쓰면 하의 분류가 퇴행할 수 있다. 다음 서비스 후보는 상의와 하의의 사람 검수 test set을 함께 만족하도록 태스크별 입력 모드 또는 분리 체크포인트를 지원한 뒤 결정한다.

## 로컬 산출물

원본 이미지, 마스크, 임베딩 캐시, 체크포인트는 라이선스와 용량 때문에 Git에 커밋하지 않는다.

| 산출물 | 로컬 상대 위치 (`work/upper-fit-data`) |
|---|---|
| 주석과 manifest | `dataset/annotations.csv`, `dataset/manifest.json` |
| 검수 시트 | `dataset/review/` |
| 임베딩 캐시 | `cache/train.pt`, `cache/val.pt`, `cache/test.pt` |
| 5개 체크포인트와 지표 | `outputs/upper_fit_<feature_mode>.pt`, `outputs/upper_fit_<feature_mode>.metrics.json` |

모든 체크포인트 메타데이터에는 `research_only_until_human_label_review`가 기록돼 있다.
