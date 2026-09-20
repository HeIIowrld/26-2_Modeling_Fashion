# web/ — 웹 애플리케이션

브라우저 화면과 서버 코드만 모아둔 폴더입니다. **웹 전용 파일은 여기 있는 것이 전부입니다.**

## 실행

저장소 루트에서 실행합니다.

```bash
python web/run_web.py
```

켜기 전에 파이썬 버전·패키지·데이터 파일을 점검하고, 문제가 있으면 무엇을 고쳐야 하는지 알려줍니다.
서버를 켜지 않고 점검만 하려면 `python web/run_web.py --check`.

| 상황 | 명령 |
| --- | --- |
| 포트 지정 | `python web/run_web.py --port 9000` |
| 다른 기기에서 접속 | `python web/run_web.py --lan` |
| 브라우저 자동 실행 끄기 | `python web/run_web.py --no-browser` |

### 운영 배포 (`192.168.0.110`)

운영 홈페이지는 이 컨테이너의 `0.0.0.0:8000`에서 실행하고, `/api`만
비공개 SSH 터널을 통해 Slurm GPU worker로 전달합니다. GPU 모델 패키지를
이 컨테이너에 설치할 필요가 없습니다.

```bash
systemctl status fitta-web.service fitta-gpu-tunnel.service
systemctl restart fitta-gpu-tunnel.service fitta-web.service
journalctl -u fitta-web.service -u fitta-gpu-tunnel.service -f
```

`afsd.iptime.org:80`은 라우터에서 `192.168.0.110:8000`으로 포워딩한다.
GPU 노드 자체를 외부 80번에 노출하지 않는다. 자세한 구조와 Master Node
운영 명령은 `gpu_server/README.md`를 본다.

### 모델 없이 화면 전체 흐름 확인

모델 패키지가 없는 컨테이너나 디자인 검토 환경에서는 표준 라이브러리만 쓰는
목업 서버를 실행할 수 있습니다. 정적 화면뿐 아니라 사진 업로드, 분석 진행 상태,
추천 결과, 예상 착장샷, 피드백, 삭제 API까지 동일한 경로로 시연합니다.

```bash
python3 web/mock_server.py --host 0.0.0.0 --port 8000
```

목업 응답은 실제 모델 추론 결과가 아니며 서버를 종료하면 모두 사라집니다.

## 파일

| 파일 | 하는 일 |
| --- | --- |
| `run_web.py` | 실행 진입점. 환경 점검 후 uvicorn 기동 |
| `app.py` | FastAPI 라우트, 업로드 처리, 세션 사진 보관·삭제 |
| `gateway.py` | `.110` 홈페이지 서빙과 GPU API 프록시 |
| `pipeline.py` | 분석 모듈을 Notebook과 같은 순서로 호출 |
| `requirements-gateway.txt` | 로컬 게이트웨이 최소 의존성 |
| `systemd/` | 로컬 웹·GPU 터널 시스템 서비스 |
| `static/` | 화면 (`index.html`, `lookbook.css`, `app.js`) |

## 이 폴더 밖에 있는 것

체형·의류 분석과 추천 규칙은 **웹 전용이 아니라 `main.ipynb`와 공용**이라
`ai_fashion_recommender/`에 있습니다. `pipeline.py`가 거기서 가져다 씁니다.

- 분석: `pose_analyzer.py`, `clothing_parser.py`, `outfit_analyzer.py`, `body_shape.py` 등
- 추천: `recommendation_engine.py`, `fashion_rules.py`, `product_catalog.py`
- 데이터·모델: `data/products.csv`, `FASHION_RULES_MASTER.md`, `models/`

웹 화면만 고칠 때는 이 폴더만 보면 되고, 추천 결과 자체를 바꾸려면
`ai_fashion_recommender/`를 봐야 합니다.

## 업로드한 사진

저장소 안에 두지 않습니다. OS 임시 폴더(`%TEMP%/fitta_web_sessions/`)에 저장하고
30분 뒤 자동 삭제합니다. 프로젝트가 OneDrive 같은 동기화 폴더 안에 있어도
전신사진이 클라우드로 올라가지 않게 하기 위해서입니다.

## 테스트

공용 모델 테스트와 웹 전용 테스트를 각각 실행합니다.

```bash
cd ai_fashion_recommender && python -m unittest discover -s tests -t .
cd ../web/tests && ../../.venv-web/bin/python -m unittest discover -s .
```

`ai_fashion_recommender/tests`는 파이프라인 호환성을, `web/tests`는 게이트웨이,
무신사 실시간 검색·상품 VTON과 UI 계약을 검사합니다.

## 무신사 코디 조합 추천과 다중 착장샷

분석이 끝나면 사진·사용자 조건에서 만든 키워드로 카테고리별 후보를 넓게 검색합니다.
그 뒤 `FASHION_RULES_MASTER.md`의 상·하의 조화 규칙, 검색 적합도, 신발과 선택 스타일의
관계를 함께 점수화해 세 개의 코디 조합으로 다시 정렬합니다. 사용자가 고르지 않은 상의·하의는
사진 속 현재 아이템으로 고정되므로, 상의와 신발만 바꿀 때는 현재 하의에 연결되는 상의 1개와
신발 1개가 각 LOOK에 묶입니다.

- `GET /api/jobs/{job_id}/images/{name}`: 준비된 JPEG 확인·다운로드
- `POST /api/jobs/{job_id}/tryon-products`: 무신사 카드에서 고른 상의·하의 조합 합성
- `POST /api/jobs/{job_id}/shopping-tryon-batch`: 추천 코디 3개의 배치 시작·현재 상태 반환
- `GET /api/jobs/{job_id}/shopping-tryon-batch`: 조합별 `queued/running/done/failed` 상태 조회

화면은 배치 진행률과 준비된 무신사 상품 조합을 바로 갱신하고, 결과 전환과 JPEG
다운로드를 제공합니다. 결과와 원본은 기존과 같이 30분 안에 삭제됩니다.

무신사 검색 결과가 로컬 카탈로그 상품과 일치하면 저장된 상품 이미지를 그대로 쓰고,
실시간 검색 상품은 허용된 무신사 이미지 CDN에서 현재 세션으로 안전하게 받아 합성합니다.
새 응답의 `shopping_outfits`가 있으면 카테고리 카테시안 곱을 만들지 않고 추천된 세 조합만
자동 생성합니다. 구형 결과에는 기존 카테고리 조합 방식이 호환 경로로 남아 있습니다. 완성된
결과는 배치 종료 전부터 화면에 추가되며 사용자가 전환·다운로드할 수 있습니다. 특정 상품을 선택해 해당 조합을
우선 요청할 수도 있고, 이미 생성된 조합은 캐시를 재사용합니다. CatVTON과 현재 파서는
상·하의 마스크만 지원하므로 신발은 하의에 잘못 덮어쓰지 않고 미지원으로 표시합니다.

## 현재 코디 점수와 추천 이유

`내 착장 분석` 상단에는 `FASHION_RULES_MASTER.md`를 실행하는 기존 진단기로 계산한
상의·하의 × 체형·상황·스타일의 2×3 점수표가 표시됩니다. 각 칸의 통과 기준은 85점이며,
종합 점수와 현재 착장 설명 세 문장도 함께 제공합니다.

각 LOOK에는 현재 유지하는 아이템, 실루엣·색상 조화 규칙, 실제 매칭 키워드 중 최대 3개의
검증된 근거가 표시됩니다. 조합 순위는 항상 규칙 기반 코드가 결정하고 LLM은 순위를 바꾸지
않으며, 서버가 확정한 조합 근거만 자연스러운 한 문장으로 다듬습니다.

개별 무신사 상품 선택 근거도 실제 상품명과 매칭된 키워드 중 적용 규칙·입력 출처·분석 신뢰도가
확인된 근거만 사용합니다. 근거가 없으면 설명을 임의로 만들지 않습니다. 발표 서버에서
Gemini 문장 생성을 켜려면 다음 환경 변수를 지정합니다. LLM에는 원본 사진, 예산, 사용자
목적, 전체 검색 조건을 보내지 않고 서버가 확정한 상품별 근거를 최대 3개만 보냅니다.
호출 실패나 근거 ID 검증 실패 시 같은 근거의 규칙 기반 문장으로 자동 대체합니다.

```bash
cp .env.example .env
# .env를 열어 GEMINI_API_KEY 등호 뒤에 새 키를 한 번만 입력
python web/run_web.py
```

`.env`는 서버 시작 때 자동으로 읽으며 Git에서 제외됩니다. 셸에 이미 설정된 환경변수가
있으면 셸 값을 우선합니다. API 키 값은 로그에 출력하지 않습니다. Gemini는 고객에게
직접 옷을 권하는 옷가게 점원처럼 검증된 근거를 자연스러운 한 문장으로 다듬습니다.
응답에는 실제 사용한 근거 ID를 함께 반환해야 하며, 상품명과 매칭된 키워드를 포함하지
않거나 허용되지 않은 근거 ID를 사용하면 폐기합니다. 예산·가격·할인·가성비·사이즈는
설명하지 않고, 신발에는 체형이나 다리 길이를 근거로 사용하지 않습니다.

OpenAI를 사용할 때는 `FASHION_LLM_PROVIDER=openai`, `OPENAI_API_KEY`, 원하는
`FASHION_LLM_MODEL`을 지정합니다. 브라우저 입력란에서 받은 사용자 키는 저장하거나
로그에 남기면 안 되므로, 차후 사용자별 연동은 세션 메모리 보관 또는 Google OAuth로
별도 구현하는 것을 전제로 합니다.
