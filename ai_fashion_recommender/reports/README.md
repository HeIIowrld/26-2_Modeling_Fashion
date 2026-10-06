# 실험 보고서 색인

실행 모듈은 `../src/`, 학습·평가 CLI는 `../scripts/`와 `../experiments/`에 있습니다. 이 폴더는 실험 결과와 판단 근거를 보존합니다. 날짜가 있는 보고서는 해당 시점의 기록입니다.

| 분야 | 먼저 읽을 보고서 | 원자료 |
| --- | --- | --- |
| 의류 속성 헤드 개선 | [최종 채택 결과](FINAL_REPORT.md), [3차 보강 미채택 판단](ROUND3_FINAL_REPORT.md) | 번호가 붙은 JSON 평가 기록 |
| 학습 데이터 출처·분할 | [출처 기록 안내](manifests/README.md) | `manifests/`의 CSV·JSON |
| 상품 사진 속성 | [C1 평가](PRODUCT_ATTRIBUTE_C1.md), [C2 평가](PRODUCT_ATTRIBUTE_C2.md) | `attribute_c1/`의 주석·예측·평가 기록 |
| 가상 피팅 품질 | [VTON 품질 조사](vton_quality/README.md) | `vton_quality/`의 조건별 평가와 사례 |
| 체형 보정 | [체형 사전 정보 통합](BODY_SHAPE_PRIOR_INTEGRATION.md) | 체형·사진 가림 관련 보고서 |
| 보조 핏 헤드 | [핏 검증](FIT_VISION_BOOTSTRAP_RESULTS.md), [상의 핏 검증](UPPER_FIT_BOOTSTRAP_RESULTS.md) | 각 학습·채택 기준 보고서 |

평가 수치는 데이터 분할·채택 기준과 함께 읽어야 합니다. 자동 검사 성공을 사용자 만족도나 실제 착용감으로 해석하지 않습니다.
