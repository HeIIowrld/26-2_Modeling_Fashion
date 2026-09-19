# 바디쉐입 가중치 실험 최종 보고서

## 1. 실험 목적

체형 규칙의 R-BOD 점수에 multiplier를 적용했을 때 추천 상품의 전체 순위와 Top-10/Top-3가 실제로 변하는지 확인했다. production 기본 동작은 유지하고, binary scoring과 semantic group scoring을 고정 후보군에서 비교했다.

## 2. 실험 조건

- multiplier: `0`, `1`, `1.5`, `2`, `3`
- fixed candidate: 75개
- 후보 수집 및 이미지 분석: 재실행하지 않음
- 사용 artifact: 기존 candidate cache, scores.csv, body_shape_connections.json
- ranking: category별 기존 정렬 후 interleave 유지
- pilot 범위: 동일인 1명 × 3장 (`baseline_front`, `arm_pose`, `slight_angle`)

### 체형별 R-BOD 연결

| condition | 체형 | R-BOD 연결 |
|---|---|---|
| `baseline_front` | 모래시계체형 | `R-BOD-03` |
| `arm_pose` | 둥근체형 | `R-BOD-07` |
| `slight_angle` | 둥근체형 | `R-BOD-07` |

코드와 규칙에는 마름모꼴체형 `R-BOD-08`도 연결되어 있으나, 이번 3장 pilot 입력에는 해당 condition이 포함되지 않았다.

## 3. Binary와 semantic group 비교

### Binary

- `baseline_front`: body_shape_score `0:58`, `4:17`
- `arm_pose`: `0:21`, `4:54`
- `slight_angle`: `0:21`, `4:54`
- multiplier `0`에서 전체 rank 변화는 각각 `23`, `29`, `29`개였다.
- multiplier `1.0`, `1.5`, `2.0`, `3.0`에서는 Top-10과 Top-3 변화가 없었다.

### Semantic group

- `baseline_front`: `0:58`, `4:17` (unique 2개)
- `arm_pose`: `0:21`, `1.333:37`, `2:17` (unique 3개)
- `slight_angle`: `0:21`, `1.333:37`, `2:17` (unique 3개)
- `arm_pose`와 `slight_angle`은 점수 단계가 세분화되었지만, 모든 tested multiplier에서 전체 rank, Top-10, Top-3 변화가 없었다.
- 모든 condition과 mode에서 Top-3는 `MS3075254`, `MS5717317`, `MS6104960`으로 동일했다.
- baseline_front 대비 arm_pose/slight_angle Top-3 overlap은 모두 `1.000`이었다.

## 4. 상품명 메타데이터 한계

실제 상품명 기반 group 매칭은 다음과 같았다.

| semantic group | baseline_front | arm_pose | slight_angle |
|---|---:|---:|---:|
| `fit_silhouette` | 17/75 (22.67%) | 54/75 (72.00%) | 54/75 (72.00%) |
| `neckline_structure` | 0/75 (0%) | 0/75 (0%) | 0/75 (0%) |
| `waist_emphasis` | 0/75 (0%) | 0/75 (0%) | 0/75 (0%) |
| `length` | 0/75 (0%) | 0/75 (0%) | 0/75 (0%) |

`fit_silhouette` 외 group은 active rule이 있어도 상품명에서 검출되지 않았다. 따라서 semantic score의 세분화가 실제 상품 ranking 정보로 충분히 전달되지 않았다.

## 5. 최종 결론

이번 실험의 최종 production 결정은 **multiplier 1.0 유지**다.

판정 원인은 다음 두 가지다.

- **B. 상품명 메타데이터 부족으로 일부 group 검출 불가**
- **D. category interleave 때문에 semantic 방식에서도 변화 없음**

semantic group은 score unique 개수를 늘렸지만, 현재 fixed candidate와 category interleave에서는 Top-10/Top-3를 바꾸지 못했다. multiplier `1.5`, `2.0`, `3.0`을 적용할 실험 근거도 확인되지 않았다.

production 기본값은 다음을 유지한다.

- `body_shape_multiplier = 1.0`
- `body_shape_mode = "binary"`
- semantic group은 실험 옵션으로만 유지
- 글로벌 `fit` weight `4.0` 유지

## 6. 한계와 다음 단계

이번 결과는 동일인 1명 × 3장으로 수행한 pilot이다. 포즈와 입력 품질의 일반화 결론으로 확대할 수 없다.

다음 단계는 **정면 전신사진 입력 검증**이다. 여러 사람과 정면 전신 입력에서 체형 판정, R-BOD 연결, 상품명 metadata coverage, category interleave 이후의 ranking 변화를 다시 확인해야 한다.
