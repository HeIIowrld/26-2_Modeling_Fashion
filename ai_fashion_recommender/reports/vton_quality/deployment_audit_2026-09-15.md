# VTON 추가 수정 운영 배포

- 전환: 2026-09-15 17:47 KST.
- 이전 릴리스: `/data1/dsl01/releases/fitta_20260915_vton_quality`.
- 현재 릴리스: `/data1/dsl01/releases/fitta_20260915_vton_quality_audit`.
- 운영 링크: `/data1/dsl01/releases/fitta_current`.
- GPU 워커: Slurm 222556, `fitta-web.service` active.

이전 운영본을 복제하고 품질 검사·입력 검사·재생성 선택 코드, 평가 스크립트와 관련 테스트에 패치를 적용했다. 수정한 기존 파일 5개 외의 파일 내용 및 심볼릭 링크가 동일함을 확인했다. 팀원의 사이즈 기능과 모델 가중치를 보존했다. 추가 회귀 테스트 2개 파일과 재집계 JSON도 포함했다.

## 검증

- 서버 전체 테스트: Slurm 222555, 473 passed / 2 skipped / 669 subtests passed. 기존 라이브러리 deprecation warning 5건.
- 공개 `/api/health`: CUDA, FASHN 파서, VTON 활성화, 상품 2,224개 확인.
- 실제 사진 `049713_0.jpg`로 분석 및 합성: Slurm 222557, SMOKE_OK.
- 두 조합 `MS3988401+MS7326873`, `MS7281184+MS7326873`의 JPEG 생성·반환, 동일 조합 캐시 재사용 확인.
- 테스트 세션 삭제 확인. 동작 검증은 이미지의 미적 품질이나 핏 정확도에 대한 독립적인 평가가 아니다.
- 로그: `/data1/dsl01/eval/vton_deploy_20260915/validate_222555.log`, `smoke_222557.log`.

첫 검증 제출은 Windows 줄바꿈, 다음 제출은 pytest 경로 누락으로 실패했다. 이를 수정한 최종 검증이 통과한 뒤에만 운영 경로를 전환했다.

## 남은 범위

공개 홈페이지의 기장 입력 버튼은 아직 미반영이다. 공개 `/app.js`에 해당 함수가 없음을 확인했다. `.110` 게이트웨이의 SSH 접속 경로가 필요하다. 서버 API와 이번 입력·합성 품질 검사 수정은 운영 반영됐다. PR #23 병합은 수행하지 않았다.

## 되돌리기

이전 릴리스를 보존했으므로 필요할 때 다음 명령으로 복구할 수 있다. 이번 배포 중에는 실행하지 않았다.

```bash
ln -sfn fitta_20260915_vton_quality /data1/dsl01/releases/fitta_current
systemctl --user restart fitta-web.service
```
