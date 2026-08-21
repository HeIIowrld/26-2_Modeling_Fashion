# 인수인계: 생성모델(CatVTON) 튜닝 세션

2026-08-21 기준. 새 세션에서 VTON 합성 품질을 손볼 때 먼저 읽는 문서다.

## 실행 환경

```
python : C:\Users\jeff4\vton\Scripts\python.exe     # ROCm 전용 venv
GPU    : AMD RX 9070 XT (gfx1201), torch+rocm
테스트 : cd ai_fashion_recommender && <위 python> -m pytest tests/ -q
         현재 199 passed (+596 subtests), 약 90초
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

다만 이후 `body_shape.py` / `body_measure.py`가 둘레 기반으로 재설계됐으므로
**이 두 항목은 현재 코드에서 다시 측정해야 한다.** 위 수치는 재설계 이전 것이다.

## 이미 적용된 VTON 개선 (튜닝 아님)

전부 입출력 처리이고 모델 가중치는 건드리지 않았다.

| 개선 | 내용 |
| --- | --- |
| 경계 halo 제거 | repaint 블렌딩 밴드를 마스크 안쪽으로 침식 (`repaint_inset`) |
| 세로 긴 사진 잘림 | `pad_to_aspect` / `unpad_result`로 레터박스 처리 |
| 레퍼런스 품질 지표 | `evaluate_garment_reference` (coverage / fill / contrast) |
| 기장 gap 경고 | `bottom_length_gap` / `sleeve_length_gap`, 2단계 이상이면 경고 |
| 레퍼런스 정제 | 가방·머리카락·팔 등 가림 요소를 옷 마스크에서 제외 |

경고는 `generate()`에 `context={"outfit": ..., "classifier": ...}`를 넘기면
`tryon.last_warnings`에 쌓인다.

## 알려진 실패 유형

| 유형 | 상태 |
| --- | --- |
| 원래 옷보다 짧은 옷 → 마스크가 옷 텍스처로 채워짐 | **원인 확인.** gap 1 정상 / gap 3 실패. 감지만 가능 |
| 여성 상의 시스루 | **원인 미확정.** 가설 5개 기각(마스크 구멍·coverage·contrast·흰 배경·레퍼런스 자체). 시드 무관 |
| 아우터(롱코트·재킷) 착용 시 붕괴 | 원래 옷 실루엣이 남는다. 롱코트가 롱스커트로 변한 사례 |
| 크로스백이 흰 덩어리로 뭉개짐 | 미조사 |
| 허리춤에 없던 카키색 띠 | 미조사 |

시스루 조사 기록은 `reports/vton_quality/README.md`에 있다. 기각한 가설을 다시
세우지 않도록 먼저 읽을 것.

## 튜닝 전에 검토할 만한 대안

파인튜닝은 위 제약 때문에 비용이 크다. 더 싼 수단부터 확인하는 편이 낫다.

1. **추론 파라미터** — `num_inference_steps`, `guidance_scale`, `pipeline_mask_blur`,
   `repaint_blur_divisor`. 현재 값은 CatVTON 기본값 기준이고 계통적으로 튜닝한 적이 없다.
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
# 배치 평가 (30장, 약 25분) — 대조 시트까지 생성
<vton python> scripts/batch_eval.py 15
```

배치 평가 스크립트는 `scripts/batch_eval.py`에 있다.
바꾸기 전후를 **같은 시드·같은 사진**으로 비교해야 판단이 선다.

## 정리되지 않은 것

- `web/`이 저장소 루트와 `ai_fashion_recommender/` 양쪽에 있다. 중복인지 확인 필요.
- 배치 평가 수치는 체형 로직 재설계 이전 것이라 갱신이 필요하다.
