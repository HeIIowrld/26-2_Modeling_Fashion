# 외투 제거 후 상의 착용: 자체 영역 처리 실험

상태: 연구용. 운영 VTON 기본 경로에는 연결하지 않았다.

## 문제와 구현

두꺼운 외투의 넓은 마스크를 새 상의 마스크로 그대로 쓰면 새 옷도 커지거나
원래 외투의 일부가 남는다. `src/outerwear_top_tryon.py`는 외투 전체를
지울 `erase_mask`와 포즈에서 추정한 좁은 `target_body_mask`를 구분한다.
목표 영역은 숨겨진 신체의 정답이 아니며, 단독으로 자연스러운 생성 품질을
보장하지 않는다.

외부 생성 모델의 결과를 그대로 최종 이미지로 채택하지 않고,
`scripts/composite_layered_source_zones.py`에서 세 원본 구역을 따로 보호한다.

1. 관측된 얼굴·머리·손 등의 인물 픽셀
2. 실제로 보이는 바지 픽셀. 새 상의가 가리는 부분은 생성 결과의 상의
   라벨을 기준으로 잠금을 해제한다.
3. 외투 제안에서 멀리 떨어진 밝고 균일한 스튜디오 배경

합성에 사용한 인물 파싱은 추정 결과다. 정답 분할 마스크는 이 알고리즘의
입력에 사용하지 않았다. `scripts/composite_layered_vton.py`에는 생성
모델의 패딩된 캔버스를 원본 좌표로 되돌리는 로직을 분리했다.

외투의 끈처럼 본체 파싱 밖에 있는 부속물은
`scripts/eval_coat_boundary_grabcut.py`에서 별도 성분으로 제안한다.
`scripts/restore_accessory_background.py`는 밝고 균일한 배경이라는
조건이 맞을 때에만 끈 주변의 생성 배경색을 행별 원본 배경에 맞춘다.
두 기능 모두 조건부 연구 처리이며 전역 마스크 팽창으로 사용하지 않는다.

## 확인된 효과와 한계

| 사례 | 확인된 변화 | 남은 결함 |
|---|---|---|
| 데님 재킷 `05047` | 기존 흰 이너를 지우고 검정 상의를 생성. 2단계 뒤 원본 잠금 시 공식 바지 픽셀 99.5811% 보존 | 위장무늬 소매가 회색으로 단순화. 원본 잠금이 상의 밑단에 밝은 이음새를 만들기도 함 |
| 롱코트 `01582` | 1단계의 새 상의 안에 보이던 원본 청바지 패치를 가림 인식 잠금으로 제거. 외투 본체+끈 제안의 이 한 장 IoU는 0.85496→0.93152 | 머리 옆 검은 조각과 끈 위치의 배경 윤곽이 남음 |
| 털 패딩 `05412` | 어깨 바깥의 넓은 흰 후광 감소 | 옛 패딩의 파란 지퍼와 청바지 접합 결함이 남음 |
| 퍼 `07123` | 바지/상의 연결 일부 개선 | 흰 외곽 후광과 주머니 안 손의 찢긴 경계가 남음 |

비교에서는 2단계 CatVTON의 상품·시드 42·50스텝·마스크 정책을 고정하고
1단계 원본 합성 정책만 바꿨다. 롱코트와 퍼에 사용한 흰 상품은 원래
긴 튜닉이다. 그 기장과 왼쪽 주머니 자체는 과대 생성이나 외투 잔상으로
판정하지 않는다. 반면 패딩의 파란 지퍼는 상품 사진에 없고 명백한 잔상이다.
공식 분할 마스크가 있는 데님·롱코트에만 해당 정량 수치를 썼으며,
퍼·패딩의 분할 품질은 육안 비교에 그친다. 원본·상품 이미지는 데이터
사용 조건 때문에 저장소에 포함하지 않았다.

일괄 5% GrabCut 외곽 확장은 롱코트 한 장에서는 도움이 됐지만 데님에서는
외투 IoU 0.87367→0.83138로 떨어지고 안쪽 상의 오인이 크게 늘었다.
따라서 전역 규칙으로 채택하지 않았다. 원본 바지 보존율 역시 화질이나
상품 충실도와 같지 않다. 현재 네 사례 모두 결함이 남아 자동 적용을
승인할 근거는 없다.

다음 우선순위는 외투 지퍼·털 후드와 실제 머리카락을 구분하는 주석과
검출, 주머니에 가려진 손의 복원 가능성 판정, 상품 소매·기장의 정합
검사다. 외부 생성 모델은 후보 생성에 쓰고, 여기의 자체 영역 처리와
실패 감지는 독립적으로 평가·개선한다.

## 검증

```bash
python -m unittest discover -s ai_fashion_recommender/tests -p 'test_outerwear_top_tryon.py'
python -m unittest discover -s ai_fashion_recommender/tests -p 'test_composite_layered_vton.py'
python -m unittest discover -s ai_fashion_recommender/tests -p 'test_composite_layered_source_zones.py'
python -m unittest discover -s ai_fashion_recommender/tests -p 'test_eval_coat_boundary_grabcut.py'
python -m unittest discover -s ai_fashion_recommender/tests -p 'test_restore_accessory_background.py'
```

2026-09-25에 위 다섯 묶음 47개 테스트가 통과했다. 실제 인물 사진
4장의 나란히 비교와 중간 생성물은 비공개 로컬 검토 자료로 보관했다.
