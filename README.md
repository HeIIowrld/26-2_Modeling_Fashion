# FITTA : 내일 뭐 입지?

**연세대학교 Data Science Lab · 2026-2 모델링 프로젝트 · CV 팀**

전신사진에서 체형과 현재 착장을 분석하고, 사용자 조건과 유지할 옷을 고려해 실제 상품으로 코디를 추천하는 AI 서비스입니다. 추천 이유와 가상 착용 이미지를 함께 제공해 코디의 전체적인 분위기를 확인할 수 있도록 합니다.

**[최종 발표자료 PDF](docs/presentations/fitta-final-presentation.pdf)** · **[개발·실험 문서](docs/README.md)** · **[웹 실행 안내](web/README.md)**

## 팀 소개

| 기수 | 이름 | 역할 |
| --- | --- | --- |
| 15기 | 최현 | 팀장 |
| 15기 | 박현진 | 팀원 |
| 16기 | 이주아 | 팀원 |
| 16기 | 조규홍 | 팀원 |

## Project Overview

온라인에서 옷을 고를 때는 상품 사진만으로 현재 옷과의 조화나 전체 착장 분위기를 판단하기 어렵습니다. FITTA는 사진 분석, 패션 규칙, 쇼핑몰 상품 탐색, 가상 피팅을 하나의 흐름으로 연결합니다.

- **체형·착장 분석:** 포즈와 의류 영역을 찾고, 현재 옷의 종류·색상·핏·기장을 분석합니다. 체형 파악용 사진을 별도로 넣을 수도 있습니다.
- **변경 범위 선택:** 상의·하의·신발 중 바꿀 품목을 선택하고 나머지 옷은 유지합니다.
- **상품 탐색·검증:** 스타일·목적·예산·색상 조건에 따라 무신사 상품을 탐색하고, 카테고리·판매 색상·실측 정보를 검증합니다. 검색 실패 시 로컬 카탈로그를 활용합니다.
- **코디 추천:** 기존 착장과 조화를 평가해 최대 3개의 LOOK과 추천 근거를 제공합니다.
- **가상 피팅:** 합성 가능한 상품만 적용하고 완성된 사진을 순차적으로 표시합니다. 입력·상품 이미지 검사와 결과 품질 검사, 조건부 재시도로 실패를 줄입니다.

## Pipeline

```text
전신사진 + 사용자 조건
        │
        ▼
입력 품질 검사 ── 부적합 사진은 재선택 안내
        │
        ▼
MediaPipe Pose / FASHN Human Parser
        │
        ▼
체형 비율 + FashionSigLIP 의류 속성 분석
        │
        ▼
패션 규칙 기반 추천 + 무신사 상품 탐색·검증
        │
        ▼
유지할 옷을 고려한 코디 조합 (최대 3 LOOK)
        │
        ▼
CatVTON 가상 피팅 → 품질 검사 → 필요한 경우 재시도
        │
        ▼
코디 설명 · 상품 링크 · 가상 착용 이미지
```

| 단계 | 주요 구현 | 사용 모델·방식 |
| --- | --- | --- |
| 사진 검증·체형 분석 | `quality_checker.py`, `pose_analyzer.py`, `body_shape.py` | MediaPipe Pose, 사진 기반 상대 비율 |
| 의류 영역 분리 | `clothing_parser.py` | FASHN Human Parser |
| 착장 속성 분석 | `fashion_attribute_model.py`, `outfit_analyzer.py` | FashionSigLIP 고정 백본 + 학습한 속성 분류 헤드 |
| 추천·상품 탐색 | `recommendation_engine.py`, `musinsa_live_search.py`, `outfit_combination_recommender.py` | 패션 규칙, 상품 조건 검증, 코디 조화 평가 |
| 가상 피팅·검사 | `catvton_tryon.py`, `tryon_quality.py` | CatVTON, 마스크·전처리 조정, 자동 검사 |
| 서비스 제공 | `web/app.py`, `web/pipeline.py`, `web/static/` | FastAPI, HTML·CSS·JavaScript |

분석·추천 모듈은 `ai_fashion_recommender/src/`에 있으며 웹과 Notebook이 공용으로 사용합니다. 신발 인식·합성 및 일부 의류 전환 편집은 별도 체크포인트와 실행 환경이 필요한 확장 기능입니다.

## Data & Models

| 데이터·모델 | 용도 | 저장 위치 |
| --- | --- | --- |
| Fashionpedia / Fashion200K 주석 | 의류 속성 헤드 학습·평가 | `ai_fashion_recommender/data/` |
| 상품 카탈로그·색상 메타데이터 | 오프라인 추천·검색 보완 | `ai_fashion_recommender/data/` |
| 패션 규칙 | 목적·실루엣·색상·날씨 등 추천 근거 | `ai_fashion_recommender/FASHION_RULES_MASTER.md` |
| 속성 분류 헤드 | 기준 모델과 채택한 보강 모델 | `ai_fashion_recommender/models/` |
| 출처·학습 분할 기록 | 데이터 중복·실험 재현 확인 | `ai_fashion_recommender/reports/manifests/` |

사진 원본, 전체 학습 이미지, 외부 모델 저장소, 모델 캐시와 실행 결과는 Git에 포함하지 않습니다. 별도로 확보한 이미지의 사용 조건은 각 출처를 따라야 합니다. 데이터 준비는 [데이터 안내](ai_fashion_recommender/data/README.md), 모델 설정은 [실행 환경 안내](ai_fashion_recommender/docs/RUNTIME_PROFILES.md)를 참고하세요.

## Output / Evaluation

### 의류 속성 헤드 개선

FashionSigLIP 백본은 고정하고 의류 속성 분류 헤드를 학습했습니다. 저장된 실험 보고서의 데이터 보강 전후 결과입니다.

| 모델 | 학습 crop 수 | Mean score | Mean macro-F1 |
| --- | ---: | ---: | ---: |
| 초기 기준 모델 | 4,789 | 0.7482 | 0.6054 |
| 채택한 2차 보강 모델 | 22,341 | 0.7641 | 0.6726 |

macro-F1은 약 6.7%p 개선됐습니다. 3차 보강 모델은 기존 검증셋 지표가 하락해 채택하지 않았습니다. 이 수치는 해당 실험의 검증 데이터 기준이며, 한국 사용자 전신사진 전체의 서비스 정확도를 의미하지 않습니다.

실험 설정·평가 분할·태스크별 결과: [속성 모델 최종 보고서](ai_fashion_recommender/reports/FINAL_REPORT.md), [3차 보강 채택 여부](ai_fashion_recommender/reports/ROUND3_FINAL_REPORT.md).

### 추천·가상 피팅 개선

최종 발표에서는 추천 범위 확장, 유지할 옷을 고려한 코디, 다중 검색어 탐색, 카테고리·판매 색상 검증, 상품 이미지 기반 핏 보완과 합성 전처리를 정리했습니다. 발표자료의 동일 검증 세트에서 자동 검사 실패율은 **44.4% → 11.1%**로 감소했습니다. 자동 검사 실패율은 실제 착용감이나 사용자 만족도 지표와 구분해야 합니다.

조건별 실험과 실패 사례는 [VTON 품질 조사](ai_fashion_recommender/reports/vton_quality/README.md)에 있습니다. 최종 발표 PDF는 전달받은 원본을 내용 변경 없이 첨부했습니다.

## Repository Structure

```text
26-2_Modeling_Fashion/
├── README.md                         프로젝트 소개·실행·제출 안내
├── docs/                             프로젝트 문서와 최종 발표자료
│   └── presentations/
│       └── fitta-final-presentation.pdf
├── web/                              FastAPI 웹 서비스와 화면
│   ├── static/                       HTML·CSS·JavaScript
│   ├── tests/                        API·화면 회귀 테스트
│   └── systemd/                      운영 서비스 설정
├── ai_fashion_recommender/            웹·Notebook 공용 모델링 코드
│   ├── src/                          분석·추천·가상 피팅 모듈
│   ├── data/                         카탈로그·학습 주석·설정
│   ├── models/                       속성 헤드·검증 메타데이터
│   ├── scripts/                      데이터 준비·학습·평가 CLI
│   ├── experiments/                  모델 개선 실험
│   ├── reports/                      평가 결과·출처 기록
│   ├── docs/                         모델·학습·환경별 사용 안내
│   ├── tests/                        모델링 회귀 테스트
│   └── main.ipynb                    단계별 실행 Notebook
├── gpu_server/                       GPU·Slurm 실행 및 배포 도구
└── .github/workflows/                Linux·Windows 테스트·정적 화면 배포
```

`datasets/`, `third_party/`, `outputs/`, `packages/`는 로컬에 준비·생성하는 폴더이며 Git 관리 대상에서 제외됩니다. 분석 모듈의 기존 경로는 유지해 웹·학습·GPU 작업의 호환성을 보존했습니다.

## Usage

아래 명령은 저장소 루트에서 실행합니다. **Python 3.11**을 권장합니다.

### 모델 없이 화면 흐름 확인

```bash
git clone https://github.com/HeIIowrld/26-2_Modeling_Fashion.git
cd 26-2_Modeling_Fashion
python web/mock_server.py --host 127.0.0.1 --port 8000
```

`http://127.0.0.1:8000`에서 업로드·추천·피팅 화면을 확인할 수 있습니다. 목업 데이터로 동작하며 실제 모델 추론은 수행하지 않습니다.

### 실제 분석·추천 실행

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r ai_fashion_recommender/requirements.txt
python web/run_web.py --check
python web/run_web.py
```

GPU 사용 시 운영체제·드라이버에 맞는 PyTorch를 먼저 설치해야 합니다. 최초 분석에는 외부 모델 다운로드가 필요합니다. CPU·별도 신발 환경은 [환경별 안내](ai_fashion_recommender/docs/RUNTIME_PROFILES.md), CatVTON·GPU 실행은 [모델링 안내](ai_fashion_recommender/README.md)와 [GPU 서버 안내](gpu_server/README.md)를 따르세요. 가상 피팅은 `FASHION_ENABLE_VTON=1`과 모델·체크포인트 준비가 필요합니다.

선택적으로 LLM 추천 설명을 사용할 때는 `.env.example`을 참고합니다. API 키나 사용자의 사진을 저장소에 커밋하지 마세요.

### 테스트

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r ai_fashion_recommender/requirements-test.txt
python -m pytest -q
```

GitHub Actions에서 Linux·Windows 회귀 테스트를 실행합니다. GPU에서 실제 합성 품질까지 평가하는 검사는 아닙니다.

## Limitations

- **체형:** 사진의 비율은 참고 정보입니다. 촬영 구도·포즈·몸선을 가리는 옷 때문에 오차가 발생할 수 있습니다.
- **사이즈·착용감:** 실측과 입력 정보가 있어도 정확한 착용감, 신축성, 개인 취향까지 보장하지 않습니다. 실측이 없는 상품도 있습니다.
- **추천:** 패션 규칙과 제한된 사용자 조건을 활용하며, 대규모 사용자 만족도 실험으로 검증한 개인화 모델은 아닙니다.
- **상품 정보:** 가격·색상·재고는 수집 시점과 외부 서비스 상태의 영향을 받습니다. 구매 전 상품 페이지를 확인해야 합니다.
- **합성:** 사진·상품 이미지·핏·기장 변경에 따라 품질 차이가 있습니다. 실제 핏을 보장하는 사진이 아닙니다.

## References

- [README 구성 참고: DSL 26-1 NLP 팀](https://github.com/DataScience-Lab-Yonsei/26-1_DSL_Modeling_NLP1)
- [MediaPipe](https://github.com/google-ai-edge/mediapipe)
- [FASHN Human Parser](https://github.com/fashn-AI/fashn-human-parser)
- [FashionSigLIP](https://huggingface.co/Marqo/marqo-fashionSigLIP)
- [Fashionpedia](https://github.com/cvdfoundation/fashionpedia)
- [Fashion200K](https://github.com/xthan/fashion-200k)
- [CatVTON](https://github.com/Zheng-Chong/CatVTON)

외부 모델·데이터는 각 프로젝트의 라이선스를 따릅니다. 저장소에 전체 모델·데이터의 재배포 권한이 포함되는 것은 아닙니다.
