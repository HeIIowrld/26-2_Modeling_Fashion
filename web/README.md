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
예산 API, 1순위 VTON 캐시와 UI 계약을 검사합니다.

## 자동 다중 착장샷

분석이 끝나면 서버가 상품이 있는 상위 추천을 최대 3개까지 자동으로 합성합니다.
1순위가 분석 단계에서 이미 실제 VTON으로 생성됐다면 그 파일을 재사용하고, 나머지는
GPU에서 순차 생성합니다. 한 순위가 실패해도 다른 순위는 계속 처리합니다.

- `POST /api/jobs/{job_id}/tryon-batch`: 배치를 중복 없이 시작하거나 현재 상태 반환
- `GET /api/jobs/{job_id}/tryon-batch`: 순위별 `queued/running/done/failed` 상태 조회
- `GET /api/jobs/{job_id}/images/{name}`: 준비된 JPEG 확인·다운로드
- `POST /api/jobs/{job_id}/tryon-products`: 무신사 카드에서 고른 상의·하의 조합 합성

화면은 배치 진행률과 준비된 순위를 바로 갱신하고, 이전/다음 렌더 전환 및 순위별
JPEG 다운로드를 제공합니다. 결과와 원본은 기존과 같이 30분 안에 삭제됩니다.

무신사 검색 결과가 로컬 카탈로그 상품과 일치하면 저장된 상품 이미지를 그대로 쓰고,
실시간 검색 상품은 허용된 무신사 이미지 CDN에서 현재 세션으로 안전하게 받아 합성합니다.
사용자는 상의와 하의를 카테고리별로 하나씩 골라 함께 입혀볼 수 있고 여러 조합 결과를
화면에서 전환할 수 있습니다. CatVTON과 현재 파서는 상·하의 마스크만 지원하므로 신발은
하의에 잘못 덮어쓰지 않고 미지원으로 표시합니다.
