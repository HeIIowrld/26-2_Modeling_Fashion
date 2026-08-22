# 인수인계: 생성모델(CatVTON) 튜닝 세션

2026-08-22 기준. 새 세션에서 VTON 합성 품질을 손볼 때 먼저 읽는 문서다.

> **2026-08-22 갱신**: 추론단 개선 4건(스케줄러 교체 fast 프리셋 · 아우터 마스크
> 재배정 · 보호 영역 복원 · 스커트 선택적 guidance)을 같은 시드 A/B로 검증해
> 기본값으로 승격했다. 30장 배치 성공 15→21, 실패 2→0, 장당 86→28초.
> 상세: [reports/vton_quality/tuning_2026-08-22.md](../reports/vton_quality/tuning_2026-08-22.md).

## 실행 환경

```
python : C:\Users\jeff4\vton\Scripts\python.exe     # ROCm 전용 venv
GPU    : AMD RX 9070 XT (gfx1201), torch+rocm
테스트 : cd ai_fashion_recommender && <위 python> -m pytest tests/ -q
         현재 202 passed (+596 subtests), 약 20초
```

ROCm 관련 함정은 [AMD_ROCM_SETUP.md](AMD_ROCM_SETUP.md)에 정리돼 있다. 요약하면
FASHN 파서(SegFormer)는 MIOpen 버그로 GPU에서 죽어 CPU로 강제되고, 디퓨전만 GPU를 쓴다.
그래서 **무거운 작업을 동시에 돌리면 경합으로 멈춘다.** 한 번 22분 행이 났었다.

## 코드 위치

| 대상 | 경로 |
| --- | --- |
| VTON 어댑터 | `src/catvton_tryon.py` |
| VTON 인터페이스 | `src/virtual_tryon.py` |
| 마스크 생성 | `src/clothing_parser.py` |
| CatVTON 원본 | `../third_party/CatVTON/` |
| 합성 대상(사람) | `../datasets/people/{men,women}` (남 50 / 여 52) |
| 합성 타겟(상품) | `../datasets/garments/{raw,clean}` |
| 검증 스크립트 | `scripts/test_tryon.py` |
| 품질 리포트 | `reports/vton_quality/` |

`ENABLE_VTON`은 `src/config.py`에서 기본 False다. 합성은 `--vton` 플래그나
Notebook의 `USE_VTON = True`로 켠다.

## 튜닝 시작 전 반드시 확인할 제약

**1. CatVTON 저장소에 학습 코드가 없다.**
`third_party/CatVTON/`에는 `inference.py`, `eval.py`, `app*.py`, `model/`만 있고
train/finetune 스크립트가 없다. 파인튜닝하려면 학습 루프를 직접 짜야 한다.

**2. 학습 데이터가 없다.**
VTON 파인튜닝에는 (사람 사진, 입힐 옷, 그 옷을 입은 정답 사진) 삼중항이 필요하다.
지금 가진 건 사람 사진 102장과 상품 이미지 592장뿐이고 정답 쌍이 없다.
VITON-HD·DressCode 같은 공개 데이터셋은 별도 라이선스 절차가 필요하다.

**3. 라이선스가 비상업이다.**
CatVTON 가중치는 CC BY-NC-SA 4.0이다. 파생 모델도 같은 조건을 따른다.

**4. 실패 원인의 상당수가 생성모델 밖에 있다.**
아래 배치 평가 결과를 먼저 볼 것.

## 측정된 현재 성능

30장(남 15 / 여 15) 배치 평가 결과다. 상세는
[reports/vton_quality/batch_eval_2026-08-21.md](../reports/vton_quality/batch_eval_2026-08-21.md).

| 판정 | 합계 |
| --- | --- |
| 성공 | 4 (14%) |
| 부분 실패 | 18 (62%) |
| 실패 | 7 (24%) |

렌더링 자체는 29/30 성공(1장은 포즈 검출 실패), 장당 약 50초.

당시 확인된 상위 병목 두 가지는 생성모델이 아니었다.

- **추천 쏠림** — 성별당 한 조합만 나왔다. 시스루가 나는 흰 셔츠가 여성 15명
  전원에게 배정돼 여성 실패율이 남성의 두 배가 됐다.
- **체형 분류 붕괴** — 29명 전원이 같은 체형으로 분류됐다.

**이 두 항목은 2026-08-21에 재측정했다** —
[reports/vton_quality/frontend_recheck_2026-08-21.md](../reports/vton_quality/frontend_recheck_2026-08-21.md).

- **체형 붕괴는 해소됐다.** 102장 전량에서 삼각 34 / 역삼각 29 / 사각 29 / 불확실 8로 갈린다.
- **추천 쏠림의 원인은 체형이 아니라 동점이었고, 고쳤다.** 최고점에 정확히 동점인
  후보가 남성 777개(71종 상의) / 여성 2,833개(98종 상의)였다. 안정 정렬이라
  **카탈로그 CSV 행 순서**가 승자를 정해 모두가 같은 첫 줄을 받았다.
  `recommendation_engine.py`의 정렬 키를 `(-총점, _tie_rank(사진 씨앗, 상품))`으로
  바꿔 동점끼리만 갈랐다. 점수와 최고점(94.72)은 그대로다.
  - 전원 기준 top-1 조합: 남 1 → **47가지**, 여 1 → **52가지**.
  - **시스루 흰 셔츠 MS6797005는 100명 중 0명에게 배정된다.** (전에는 여성 전원)

### 동점 수정 이후 배치 재평가 (2026-08-21)

같은 30장을 다시 돌렸다 — [batch_eval_after_tiefix.md](../reports/vton_quality/batch_eval_after_tiefix.md).

| 판정 | 최초 | 동점 수정 | + repaint300 | + 2026-08-22 개선 4건 |
| --- | --- | --- | --- | --- |
| 성공 | 4 (14%) | 14 (48%) | 15 (52%) | **21 (72%)** |
| 부분 실패 | 18 (62%) | 13 (45%) | 12 (41%) | 8 (28%) |
| 실패 | 7 (24%) | 2 (7%) | 2 (7%) | **0** |

마지막 열은 `outputs/batch_eval_newdefaults/`(DPM++ 25스텝 fast 경로, 장당 28초)다.
채점 세션이 다르고 스케줄러가 바뀌어 시드 고정 비교가 아니라는 주의는 동일하다.
회귀는 IMG_5410(상의 얼룩) 1장뿐이고, 개선별 인과는 같은 시드 A/B
(`outputs/vton_tuning_0822/`)로 따로 고정돼 있다.

**최초 판정은 채점자가 다르므로 폭은 그대로 믿지 말 것.** 방향은 분명하다.
뒤의 두 열은 같은 채점자·같은 사진·같은 시드라 직접 비교할 수 있다.
여성 실패율이 남성의 두 배였던 문제는 사라졌다(남 6/7/1, 여 8/6/1).
하의 기장 gap 경고도 15건 → 7건으로 줄었다.

장당 시간은 50초 → 86초로 늘었는데, 추천이 다양해져 상품 레퍼런스 캐시가 매번
새로 만들어지기 때문이다. 합성 자체가 느려진 게 아니다.

## 이미 적용된 VTON 개선 (튜닝 아님)

전부 입출력 처리이고 모델 가중치는 건드리지 않았다.

| 개선 | 내용 |
| --- | --- |
| 경계 halo 제거 | repaint 블렌딩 밴드를 마스크 안쪽으로 침식 (`repaint_inset`) |
| 세로 긴 사진 잘림 | `pad_to_aspect` / `unpad_result`로 레터박스 처리 |
| 레퍼런스 품질 지표 | `evaluate_garment_reference` (coverage / fill / contrast) |
| 기장 gap 경고 | `bottom_length_gap` / `sleeve_length_gap`, 2단계 이상이면 경고 |
| 레퍼런스 정제 | 가방·머리카락·팔 등 가림 요소를 옷 마스크에서 제외 |
| 블렌딩 밴드 축소 | `repaint_blur_divisor` 150 → 300 (2026-08-21, A/B 근거 있음) |
| 스케줄러 교체 fast | `fast()` = DPM++ 2M Karras 25스텝. DDIM 50 대비 텍스처 지표 우세·시간 절반. 원인: eta 미전달로 DDIM이 eta=1.0(확률 샘플링)이던 것 (2026-08-22) |
| 아우터 마스크 재배정 | `outerwear_policy="reassign"` 기본. 코트·재킷 감지(하드셋 7/8 적중·오탐 0) 시 힙 아래 오버행을 하의 패스로 이전. 롱코트 4장 전부 개선 (2026-08-22) |
| 보호 영역 복원 | `protect_restore=True` 기본. 가방·모자·손을 최종 결과에서 원본 픽셀로 강제 복원. 한계: 파서 오라벨(검정 가방→top)은 못 살림 (2026-08-22) |
| 스커트 선택 guidance | `skirt_guidance_scale=1.5` 기본. 스커트 레퍼런스에만 gs 하향 — 시스루·치마 슬릿 완화, 비용은 복잡 패턴 퇴색. 단조 반응 2장 재현 (2026-08-22) |

경고는 `generate()`에 `context={"outfit": ..., "classifier": ...}`를 넘기면
`tryon.last_warnings`에 쌓인다.

## 알려진 실패 유형

| 유형 | 상태 |
| --- | --- |
| 원래 옷보다 짧은 옷 → 마스크가 옷 텍스처로 채워짐 | **원인 확인.** gap 1 정상 / gap 3 실패. 감지만 가능 |
| 여성 시스루 | **완화됨(2026-08-22).** guidance 단조 반응을 2장에서 재현하고 `skirt_guidance_scale=1.5` 선택 적용. IMG_5383 해소, IMG_5534 실패→부분. 근본 원인은 여전히 미확정 |
| 아우터(롱코트·재킷) 착용 시 붕괴 | **완화됨(2026-08-22).** `outerwear_policy="reassign"` 마스크 수술로 롱코트 4장 전부 개선(IMG_5524 성공 전환, IMG_5521 실패→부분). 오버행 없는 재킷의 민소매화(IMG_5518)는 잔존 |
| 크로스백·가방 뭉개짐 | **부분 해결(2026-08-22).** `protect_restore`로 라벨된 가방·끈은 복원(IMG_5387 성공 전환). 파서가 가방을 top으로 오라벨하면(IMG_5455 검정 가방, IMG_5518 모자) 못 살리고, 마스크 안에 가방을 그리는 콘텐츠 환각(IMG_5491)도 남음 |
| 미세 패턴 뭉개짐 | 잔존. 잔줄무늬 니트가 퍼 질감으로 과장된 사례(IMG_5394, 2026-08-22) |
| dpm25 색 얼룩 | 신규 관찰 1장(IMG_5410 상의 노란 얼룩). 스케줄러 고유인지 단발인지 미확인 |

시스루 조사 기록은 `reports/vton_quality/README.md`에 있다. 기각한 가설을 다시
세우지 않도록 먼저 읽을 것.

## 튜닝 전에 검토할 만한 대안

파인튜닝은 위 제약 때문에 비용이 크다. 더 싼 수단부터 확인하는 편이 낫다.

1. **추론 파라미터** — **2026-08-21에 A/B로 훑었다**
   ([param_tuning_2026-08-21.md](../reports/vton_quality/param_tuning_2026-08-21.md),
   `scripts/tune_vton.py`). 요약:
   - `repaint_blur_divisor` 150 → **300으로 바꿨다(적용 완료).** halo가 줄고 시간 비용이 없다.
     30장 재검증에서 악화 0장, 성공 14 → 15. 판정이 바뀐 건 IMG_5413(흰 세로 띠 소멸) 한 장이고,
     그 밖에 여러 장에서 원래 옷 색 잔류가 줄었다.
   - `num_inference_steps`는 기본값(50)을 유지하고 **`CatVTONTryOn.fast()` 프리셋(25스텝)**을
     추가했다(44초 → 24.5초). `batch_eval.py --fast`로 쓴다. 75는 값을 못 한다.
   - `guidance_scale`은 **바꾸지 말 것.** 올리면 텍스처가 살고 시스루가 심해진다(트레이드오프).
     **시스루가 이 값에 단조 반응한다는 건 새 발견이다** — 기존 조사에서 시험한 적 없다.
   - `pipeline_mask_blur`는 3/9/17 사이에 측정 가능한 차이가 없다.
   - **아우터 붕괴는 10개 변종 전부에서 동일하게 남았다.** 파라미터로 못 고친다.
2. **마스크 생성 방식** — `_solidify_mask` / `_dilate_mask`의 커널 크기.
   볼록 껍질은 이미 시도했다가 망토 아티팩트로 되돌렸다(기록 있음).
3. **레퍼런스 정제** — 착용컷에서 옷만 뽑는 품질이 결과를 크게 좌우한다.
   단독컷 확보가 근본 해결이다.
4. **다른 VTON 모델** — IDM-VTON, OOTDiffusion 등. 라이선스와 Windows/ROCm 구동
   가능성을 먼저 확인할 것.

## 검증 방법

```bash
cd ai_fashion_recommender
# 소수 확인
<vton python> scripts/test_tryon.py --vton --count 2
# 배치 평가 (30장, 약 45분) — 대조 시트까지 생성
<vton python> scripts/batch_eval.py 15 --out batch_eval_새이름
# 빠른 반복용: 25스텝 프리셋 (장당 시간 절반)
<vton python> scripts/batch_eval.py 15 --fast --out batch_eval_fast
# 포그라운드 10분 제한에 걸리면 구간으로 나눈다 (중간 저장됨)
<vton python> scripts/batch_eval.py 15 --offset 0 --limit 6 --skip-existing
```

앞단(체형·추천)만 볼 거라면 합성을 건너뛰는 쪽이 훨씬 싸다.

```bash
<vton python> scripts/frontend_eval.py --pose-only    # 체형 분포, 102장 10초
<vton python> scripts/frontend_eval.py --count 8      # 인식·추천까지, 약 4분
<vton python> scripts/frontend_eval.py --goal-sweep   # 목표별 추천 다양성, 약 1분
<vton python> scripts/frontend_eval.py --goal-sweep --per-shape 100   # 전원, 약 3분
```

추론 파라미터를 비교할 때는 `scripts/tune_vton.py`를 쓴다. 앞단을 사람당 한 번만
돌리고 디퓨전만 변종 수만큼 반복한다.

```bash
<vton python> scripts/tune_vton.py --list
<vton python> scripts/tune_vton.py --images IMG_5455,IMG_5383 --variants baseline,cfg3.5 --skip-existing
```

바꾸기 전후를 **같은 시드·같은 사진**으로 비교해야 판단이 선다.
`frontend_eval.py`는 포그라운드로 돌릴 것(백그라운드에서 멈추는 사례가 있다).

## 정리되지 않은 것

- **기본 경로(50스텝 DDIM)도 DPM++로 바꿀지.** fast 경로는 2026-08-22 배치로
  전량 검증됐다. 데모 최종본 경로까지 바꾸려면 dpm50 또는 dpm25를 기본으로 한
  추가 확인이 필요하다(IMG_5410 얼룩이 스케줄러 고유인지부터).
- **파서 오라벨 액세서리** — 검정 가방이 top으로 흡수되는 케이스(IMG_5455·5518).
  protect_restore의 복원 대상 자체가 없어 세그멘테이션 쪽 개선이 필요하다.
- **오버행 없는 재킷의 민소매화**(IMG_5518) — reassign 수술이 커버하지 않는 유형.
  민소매 레퍼런스 감지와 소매 회랑 마스크를 결합하는 아이디어만 있다.
- **개인화 강도**는 여전히 도메인 결정이 남아 있다. 동점 수정은 쏠림만 없앴을 뿐,
  체형이 총점에 기여하는 상한은 아직 4.3%이고 최고점 동점이 후보의 12%나 된다.
  세 갈래(기본값에서도 체형 반영 / 체형 가중치 상향 / 점수 항목 세분화)를
  재측정 리포트에 정리해 뒀다.
- **분리된 백그라운드 프로세스로 파이썬을 띄우면 모델 로드 직후 멈춘다.** CPU 0%·스레드
  2~3개로 블록되고 GPU도 CPU도 쓰지 않는다. `frontend_eval.py`·`batch_eval.py`·
  `tune_vton.py`에서 **4회 재현**했다. 포그라운드로 돌린 같은 명령은 정상이다.
  포그라운드 실행이 시간 제한에 걸려 백그라운드로 넘어가는 순간에도 멈춘다.
  원인 미확인. **무거운 스크립트는 포그라운드로 돌리고 --offset/--limit이나
  --variants로 나눠 실행할 것.** 세 스크립트 모두 중간 저장과 --skip-existing을 지원한다.

## 정리된 것

- `web/` 중복 — **중복 아니었다.** `ai_fashion_recommender/web/`에는 `__pycache__`의
  `.pyc` 두 개만 있고 소스가 없다. 디렉토리 재구성 전 잔재이고 git에도 없다.
  실제 웹 앱은 저장소 루트 `web/` 하나뿐이다.
