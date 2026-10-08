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

## 재현 자료 보관 — 2026-10-08

[GitHub Release](https://github.com/HeIIowrld/26-2_Modeling_Fashion/releases/tag/reproduction-2026-10-08)에 로컬 재현 자료를 보관했습니다. [파일 목록·SHA-256·복구 경로](manifests/reproduction_archive_2026-10-08.json)는 Git에서 관리합니다.

| 파일 | 보관 내용 |
| --- | --- |
| `fitta-source-2026-10-08.zip` | `f5df16a8fa0186820c10d846137d42ee65e4452e`의 Git 추적 코드·문서·채택 모델 전체 |
| `fitta-reproduction-metadata-2026-10-08.zip` | 9월 검증 실행과 외투 실험의 설정·수치·기준 코드·로그 381개, 원본 바이트 그대로 |
| `ai_fashion_recommender_final_r2.zip` | 기존 145개 파일의 과거 R2 배포본. 최신 웹 코드는 위 source ZIP을 사용 |
| `PACKAGE_CHECKSUMS.json` | 과거 R2 배포본과 두 속성 헤드의 체크섬 |

9월 검증 로그는 해당 시점의 기록이며 10월 코드의 검증 결과가 아닙니다. 개인 사진, 데이터셋·상품 이미지, 브라우저 프로필과 모델·임베딩 캐시는 이 보관본에 포함하지 않고 로컬에 유지합니다. 외투 실험의 HTML·이미지·배열도 유지하며, 메타데이터 ZIP만으로 시각 검토 화면 전체를 복구할 수는 없습니다.

저장소 루트에서 다음과 같이 내려받고 모든 배포 파일의 SHA-256을 확인합니다.

```bash
gh release download reproduction-2026-10-08 \
  --repo HeIIowrld/26-2_Modeling_Fashion --dir ../fitta-reproduction-download
python - <<'PY'
import hashlib, json
from pathlib import Path
manifest = json.loads(Path("ai_fashion_recommender/reports/manifests/reproduction_archive_2026-10-08.json").read_text(encoding="utf-8"))
for asset in manifest["assets"]:
    path = Path("../fitta-reproduction-download") / asset["filename"]
    assert path.stat().st_size == asset["bytes"], path
    assert hashlib.sha256(path.read_bytes()).hexdigest() == asset["sha256"], path
print("All archive checksums match")
PY
python -m zipfile -e ../fitta-reproduction-download/fitta-source-2026-10-08.zip ../fitta-source-restored
python -m zipfile -e ../fitta-reproduction-download/fitta-reproduction-metadata-2026-10-08.zip ../fitta-reproduction-restored
```

コード는 `fitta-source-restored/fitta-source/`에 복구됩니다. 메타데이터는 `fitta-reproduction-restored/repository/`와 `fitta-reproduction-restored/outerwear_visual_review/`에 나뉩니다. 첫 폴더의 내용은 저장소 루트로, 둘째 폴더의 내용은 별도의 `outerwear_visual_review_20260925` 폴더로 옮기면 기존 상대 경로를 복구할 수 있습니다. 기록에 남은 절대 경로는 당시 서버·작업 폴더를 가리키므로 새 실행에서는 실제 준비한 입력 경로를 지정하세요.

정리 대상은 업로드한 메타데이터의 기존 로컬 사본과 `packages/`의 ZIP·체크섬입니다. GitHub에서 다운로드한 사본의 파일별 해시와 PR 머지를 확인한 뒤 삭제하며, 저장소의 코드·모델·manifest와 위 제외 파일은 유지합니다.
