# GPU 서버에서 CatVTON 돌리기 (구축 완료)

CV팀 계정 **dsl01**. 2026-08-22 환경 구축과 동작 확인을 마쳤다.

## 확인된 성능

| 단계 | 시간 |
| --- | --- |
| 포즈 분석 | 0.9초 |
| FASHN 파서 마스크 | 6.7초 |
| CatVTON 파이프라인 로드 | 8.2초 |
| **합성 (20 steps, 768×1024)** | **5.8초** |

GPU: NVIDIA RTX 6000 Ada, VRAM 51GB (합성 시 3.84GB만 사용).
기본값인 50 steps로는 합성이 약 15초 예상. 내 PC(CPU 전용)에서는 장당 수 분이었다.

## 접속

```bash
# rathole → Master Node
ssh -p 222 dsl01@afsd.iptime.org

# 교내망/VPN 직결 경로
ssh -p 37220 dsl01@165.132.80.36
```

직결 경로는 학교 밖에서 연세 VPN이 없으면 TCP 타임아웃이 난다. `222`는 rathole을
통해 Master Node(`hpcmaster`)로 연결되므로 외부에서도 사용할 수 있다. 계산 노드
`hpc-stat1`의 SSH 릴레이는 `224`다. 비밀번호는 문서나 저장소에 기록하지 않는다.

## 이 클러스터에서 걸려 넘어진 것들 (전부 확인·해결)

**1. 파티션은 `partition1` 하나뿐이다.**
`sinfo`에는 `jobs`·`brl`·`gpu`도 보이지만 dsl01 계정은 `partition1`(노드 hpc-stat1)만
쓸 수 있다. `jobs`로 제출하면 `Invalid account or account/partition combination`.

**2. 같은 저장소인데 노드마다 경로가 다르다.**
master에서는 `/data1/dsl01`, 계산 노드에서는 `/mnt/data1/dsl01`(NFS).
Slurm 지시자(`--output`, `--chdir`)는 계산 노드에서 해석되므로 **반드시 `/mnt/data1`**.
`/data1`로 쓰면 로그 파일조차 못 만들고 작업이 1초 만에 죽는다.

**3. `HOME`이 계산 노드에 존재하지 않는 경로로 상속된다.**
`HOME=/data1/dsl01`이 그대로 넘어와 캐시 생성이 실패한다. 작업 안에서 덮어써야 한다.

**4. 계산 노드에는 인터넷이 없다.** master만 된다.
모델 가중치를 미리 받아 공유 캐시에 넣고 `HF_HUB_OFFLINE=1`로 돌린다.
(오프라인 플래그를 켜두면 캐시에 없을 때 조용히 멈추지 않고 바로 실패해서 원인이 드러난다.)

**5. 계산 노드 파이썬은 3.8이고 conda·module이 없다.**
프로젝트는 3.9+ 필요. 이식 가능한 3.11을 공유 저장소에 넣어 절대경로로 부른다.
venv는 절대경로를 굽기 때문에 경로가 다른 두 노드에서 깨진다 →
venv 대신 `--target` 설치 + `PYTHONPATH`.

**6. torch 기본 설치는 CUDA 13 빌드라 GPU를 못 쓴다.**
드라이버가 535(CUDA 12.2)라 `torch 2.5.1+cu121`로 맞춰야 한다.

**7. `zhengchong/CatVTON`은 저장소 전체를 받아야 한다.**
용량을 아끼려고 필요한 하위 폴더만 받으면, CatVTON 내부가 전체 `snapshot_download`를
호출하면서 "불완전한 스냅샷"으로 거부한다.

**8. `requirements.txt`에 `diffusers`가 빠져 있다.** CatVTON이 쓰는데 목록에 없다.

## 이 폴더의 파일

```
gpu_server/
├── jobs/
│   ├── tryon.sbatch       추천 + CatVTON 합성 (GPU 필요)
│   ├── enrich.sbatch      카탈로그 속성 채우기 (CPU로도 충분)
│   └── web_service.sbatch 인식 + CatVTON 작업 API (GPU 필요)
├── scripts/
│   ├── fitta_web_controller.sh       systemd에서 Slurm 작업 제출·감시
│   ├── prefetch_models.py            모델 가중치 미리 받기 (계산 노드는 인터넷이 없다)
│   ├── verify_env.py                 CUDA·패키지·프로젝트 모듈 임포트 확인
│   ├── verify_model.py               배포 모델이 제대로 로드되는지 확인
│   ├── smoke_catvton.py              데이터셋 없이 합성만 돌려보는 최소 검증
│   ├── smoke_web_api.py              인식·추천·CatVTON API 전체 검증
│   └── scan_musinsa_categories.py    유효한 무신사 카테고리 코드 탐색
└── systemd/
    └── fitta-web.service             Master Node의 Slurm 제어 유닛
```

**합성 결과 이미지는 커밋하지 않습니다.** CatVTON 데모 인물(제3자 사진)과 무신사 상품
이미지에서 나온 파생물이라 저장소 정책(`README.md`: "저작권 문제로 이미지는 커밋하지
않고 카탈로그 CSV만 관리한다")에 걸립니다. `.gitignore`가 막고 있습니다.

경로는 계산 노드 기준(`/mnt/data1/dsl01`)으로 적혀 있습니다. **master에서 수정할 때는
`/data1/dsl01`이지만 Slurm 지시자는 그대로 두어야 합니다** — 아래 함정 2번 참고.

## 구축된 것 (전부 /data1/dsl01)

```
opt/python/cpython-3.11.16-.../   이식 가능한 파이썬 3.11
opt/uv-bin/                       uv (패키지 설치기)
site-packages/                    6.2GB — torch 2.5.1+cu121 등
hf_cache/                         9.5GB — 모델 가중치
26-2_Modeling_Fashion/            프로젝트 (+ third_party/CatVTON)
jobs/                             sbatch·검증 스크립트
logs/                             작업 로그
```

받아둔 모델: `booksforcharlie/stable-diffusion-inpainting`(5.2G),
`zhengchong/CatVTON`(1.4G), `stabilityai/sd-vae-ft-mse`(639M),
`Marqo/marqo-fashionSigLIP`(2.3G), `fashn-ai/fashn-human-parser`(245M).

## 작업 제출

```bash
ssh -p 37220 dsl01@165.132.80.36
sbatch /data1/dsl01/jobs/tryon.sbatch
squeue -u $USER
```

**파티션은 `partition1` 하나만 쓸 수 있습니다.** `sinfo`에 `jobs`·`brl`·`gpu`도 보이지만
계정 연결이 없어 `Invalid account or account/partition combination`이 납니다.
`brl`에 GPU 17장이 유휴인데도 막혀 있으니, 대기가 길면 학술부에 파티션 권한을 요청하세요.

**master 노드에는 `libGL`이 없습니다.** `cv2`를 임포트하는 코드는 계산 노드에서만
돌아갑니다. `enrich_catalog.py --rederive`가 master에서 도는 건 cv2를 쓰지 않기 때문입니다.

로그: `/data1/dsl01/logs/catvton_<JobID>.out`

## 인식 + 생성 GPU 작업 서버

`web_service.sbatch`는 FASHN/FashionSigLIP 인식과 CatVTON 생성을 GPU 한 장에서
같이 실행한다. 인식 결과의 의류 마스크는 메모리에 보관했다가 최초 미리보기와
순위별 추가 합성에 그대로 재사용한다.

Master Node의 사용자 systemd 서비스가 비공개 GPU API Slurm 작업을 제출·감시한다.
실제 GPU 프로세스는 스케줄러가 할당한 계산 노드에서만 실행된다.
서비스 최초 설치는 다음과 같다.

```bash
mkdir -p ~/.config/systemd/user
ln -sfn /data1/dsl01/releases/fitta_current/gpu_server/systemd/fitta-web.service \
  ~/.config/systemd/user/fitta-web.service
loginctl enable-linger "$USER"
systemctl --user daemon-reload
systemctl --user enable --now fitta-web.service
```

운영 명령은 다음과 같다.

```bash
systemctl --user status fitta-web.service
systemctl --user start fitta-web.service
systemctl --user stop fitta-web.service
systemctl --user restart fitta-web.service
journalctl --user-unit fitta-web.service -f

# systemd가 현재 관리하는 Slurm Job ID
cat ~/.local/state/fitta-web/job_id
squeue -u "$USER"
```

`enable-linger` 설정 때문에 SSH 세션이 종료되어도 서비스가 유지된다. Slurm
작업이 시간 제한이나 오류로 종료되면 systemd가 15초 후 새 작업을 제출한다.
`stop`은 현재 Slurm 작업도 함께 취소하고, `restart`는 기존 작업을 취소한 뒤
새 Job ID로 제출한다.

이 API는 공개 홈페이지가 아니다. 계산 노드의 `0.0.0.0:8000`에서 실행하고,
`192.168.0.110` 컨테이너의 비공개 SSH 터널을 통한 요청만 받는다. GPU 노드에
외부 80번을 매핑하지 않는다.

기본 Slurm 실행 시간은 8시간이다. 로그는
`/data1/dsl01/logs/fitta_web_<JobID>.out`이다. 서비스로 실행한 작업은 직접
`scancel`하기보다 `systemctl --user stop fitta-web.service`로 종료한다. 웹 패키지는 모델 패키지와 분리한
`/data1/dsl01/web-site-packages`에서 읽는다.

## 공개 웹 게이트웨이 (`192.168.0.110`)

홈페이지와 외부 API는 GPU 서버가 아니라 `.110` 컨테이너에서 실행한다.
구조는 다음과 같다.

```text
afsd.iptime.org:80
  → 라우터/NAT → 192.168.0.110:8000 (fitta-web, 홈페이지 + API gateway)
  → 127.0.0.1:18000 (fitta-gpu-tunnel, 로컬에만 바인딩)
  → rathole SSH 222 → hpcmaster → hpc-stat1:8000 (Slurm GPU worker)
```

로컬 서비스 유닛은 `web/systemd/`, 게이트웨이는 `web/gateway.py`,
최소 의존성은 `web/requirements-gateway.txt`에 있다. `.110` 컨테이너에서 운영한다.

```bash
systemctl status fitta-web.service fitta-gpu-tunnel.service
systemctl start fitta-web.service fitta-gpu-tunnel.service
systemctl stop fitta-web.service fitta-gpu-tunnel.service
systemctl restart fitta-gpu-tunnel.service fitta-web.service
journalctl -u fitta-web.service -u fitta-gpu-tunnel.service -f
```

`fitta-web` 서비스는 `0.0.0.0:8000`에 바인딩하지만, GPU 터널의 `18000`번은
`127.0.0.1`에만 바인딩한다. 따라서 외부에 열어야 하는 포트는 `.110:8000`뿐이다.

## 상품 사진과 카탈로그 (구축 완료)

2026-08-30 서버 기준으로 `products_musinsa_enriched.csv` 2,224개 상품과
`datasets/garments/raw/` 상품 이미지 2,292장이 준비되어 있다.

`src/catvton_tryon.py:866` 이 `product.image_path` 로 옷 사진 파일을 찾는데,
저장소의 `data/products.csv`(80개)에는 **이미지 칼럼이 아예 없다**. 색상·소재·핏 같은
텍스트 속성만 있는 수작업 카탈로그라, 추천은 나와도 입힐 옷 사진이 없다.

`scripts/musinsa_crawler.py` 가 이 구멍을 메운다 — 무신사 상품 사진을
`datasets/garments/raw/` 에 받고, `image_url`·`image_path` 칼럼이 들어간
`data/products_musinsa.csv` 를 따로 쓴다.

### 다시 구축할 때 필요한 것

| 경로 | 무엇 | 필수 |
| --- | --- | --- |
| `ai_fashion_recommender/data/products_musinsa_enriched.csv` | 이미지·규칙 속성이 있는 상품 카탈로그 | **필수** |
| `datasets/garments/raw/` | 상품 대표 사진 | **필수** |
| `datasets/people/{men,women}/` | 테스트용 인물 사진 | 선택 (없으면 `data/input_person.jpg` 사용) |

`datasets/garments/clean/` 은 파서가 자동 생성·캐시하므로 받을 필요 없다.

### 복구할 때

`config.resolve_catalog()`가 enriched 카탈로그를 자동 선택하므로
**설정을 바꿀 필요 없이 파일만 넣으면 된다.**

```bash
scp -P 37220 -r datasets dsl01@165.132.80.36:/data1/dsl01/26-2_Modeling_Fashion/
scp -P 37220 ai_fashion_recommender/data/products_musinsa_enriched.csv   dsl01@165.132.80.36:/data1/dsl01/26-2_Modeling_Fashion/ai_fashion_recommender/data/
```

웹 앱은 `config.resolve_catalog()`를 사용하므로 `FASHION_PRODUCTS_CSV`가 지정되면
그 파일을 최우선으로 읽고, 없으면 enriched → 수작업 카탈로그 순서로 대체한다.

## 배포 모델

웹 서비스는
`ai_fashion_recommender/models/fashion_attribute_heads_augmented.pt`를 명시적으로
사용한다. 현재 배포본은 22,341 crop으로 학습한 augmented 헤드이며,
`/api/health`의 `trained_heads: true`로 로드 여부를 확인할 수 있다.

## 라이선스

CatVTON 가중치는 **CC BY-NC-SA 4.0(비상업)**. 발표·배포 범위를 정할 때 확인할 것.
