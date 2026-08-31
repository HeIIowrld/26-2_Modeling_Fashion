# 체형·목적 기반 AI 코디 추천

전신사진을 넣으면 체형과 지금 입은 옷을 분석하고, 패션 규칙에 맞는 상품을 골라
실제로 입혀본 합성 사진까지 만든다. DSL 2026-2 모델링 프로젝트.

```
사진 → 체형·착장 분석 → 규칙 기반 추천 → 가상 피팅
```

## 폴더 구조

```
├── ai_fashion_recommender/   코드 일체 (자세한 내용은 그 안 README)
├── datasets/
│   ├── people/               합성 대상: 옷을 갈아입힐 사람 사진
│   │   ├── men/  women/
│   └── garments/             합성 타겟: 입힐 상품 이미지
│       ├── raw/              무신사 대표컷 원본
│       └── clean/            파서로 옷만 남긴 정제본 (자동 생성)
├── collect/                  유튜브 프레임 수집 파이프라인 (현재 미사용)
├── third_party/CatVTON/      가상 피팅 모델 (직접 클론)
└── packages/                 로컬에서 생성하는 배포용 zip (Git 제외)
```

이미지 자산은 `datasets/` 한 곳에 모아 두고 코드에서는 `config.py`의
`PEOPLE_DIR` / `GARMENT_RAW_DIR` / `GARMENT_CLEAN_DIR`로만 접근한다.
저작권 문제로 이미지는 커밋하지 않고 카탈로그 CSV만 관리한다.

## 파이프라인

**1. 사용자 입력** — 전신사진, 키·체중(선택), 원하는 스타일, 코디 목적(데일리·데이트·출근·여행),
예산, 유지하고 싶은 옷. 사진은 정면 전신에 몸과 옷의 경계가 뚜렷해야 한다.

**2. 체형·자세 분석** — MediaPipe Pose로 어깨·골반·무릎·발목을 찾아 어깨/골반 비율,
상하체 비율, 다리 길이 비율을 계산한다. 사진만으로 실제 치수를 재는 건 불가능하므로
절대값이 아니라 추천용 참고값으로만 쓴다.

**3. 착장 분석** — FASHN Human Parser로 옷 영역을 픽셀 단위로 나누고,
FashionSigLIP 위에 얹은 학습된 속성 헤드로 종류·소매·기장·핏·패턴·소재를 읽는다.
마스크에서 잰 값과 학습 모델의 판정이 다르면 근거가 강한 쪽을 택한다.

**4. 추천** — `FASHION_RULES_MASTER.md`의 R-* 규칙(50개 중 43개 구현)으로 후보를 거르고
점수를 매긴다. 왜 추천했는지 근거를 함께 낸다.

**5. 가상 피팅** — CatVTON으로 추천한 옷을 실제로 입힌 사진을 만든다.
합성이 깨질 조건(레퍼런스 해상도, 원래 옷보다 짧은 기장)은 미리 감지해 경고한다.

## 데이터

**사람 사진** — `datasets/people/{men,women}` (남자 50장, 여자 52장). 인스타그램에서 모은
전신 코디 사진이다. 스크린샷이라 오른쪽에 좋아요·댓글 UI가 있고 옷 위에 텍스트가
겹친 것도 있다. 합성 결과에 글자가 남으면 원본부터 확인할 것. 연구·검증용으로만 쓰고
재배포하지 않는다.

**상품 이미지** — `datasets/garments/raw` (`scripts/musinsa_crawler.py`로 수집).
`clean/`은 파서로 옷만 남긴 정제본이고 합성 시 자동 생성·캐시된다.

**유튜브 프레임** — `collect/`에 룩북 영상에서 정면 전신 프레임을 뽑는 파이프라인이
남아 있다. 2026-08-20에 인스타 수집본으로 갈아타면서 받아둔 프레임은 지웠다.
다시 필요하면:

```bash
pip install -r collect/requirements.txt
python collect/setup.py
python collect/discover_channels.py --verify --write
python collect/collect.py --target 50
python collect/qa.py --move
```

## 시작하기

```bash
cd ai_fashion_recommender
pip install -r requirements.txt
python scripts/test_tryon.py --vton --count 2   # 수집 사진으로 바로 확인
python app.py                                   # 웹 앱 (localhost:7860)
```

`main.ipynb`를 위에서 아래로 실행해도 된다. 자세한 설정과 학습 방법은
[ai_fashion_recommender/README.md](ai_fashion_recommender/README.md) 참고.

## 알아둘 점

- 사진에서 계산한 체형 값은 실제 신체 치수가 아니다.
- 패션 규칙은 문헌 조사 기반이며 전문가 합의나 사용자 실험을 거치지 않았다.
- 속성 헤드는 공개 데이터(Fashionpedia, Fashion200K)로 학습해 한국 사용자
  전신사진에서의 성능은 별도 검증이 필요하다.
- CatVTON 가중치는 CC BY-NC-SA 4.0(비상업) 라이선스다.
- 합성 품질의 알려진 한계와 실험 기록은
  [reports/vton_quality/](ai_fashion_recommender/reports/vton_quality/)에 정리했다.
