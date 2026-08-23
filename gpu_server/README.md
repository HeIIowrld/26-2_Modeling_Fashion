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
ssh -p 37220 dsl01@165.132.80.36
```

**학교 밖에서는 연세 VPN이 켜져 있어야 한다.** 없으면 TCP 타임아웃이다.
SSH 키는 등록해 두었으므로 비밀번호를 묻지 않는다.

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
│   ├── tryon.sbatch      추천 + CatVTON 합성 (GPU 필요)
│   └── enrich.sbatch     카탈로그 속성 채우기 (CPU로도 충분)
└── scripts/
    ├── prefetch_models.py            모델 가중치 미리 받기 (계산 노드는 인터넷이 없다)
    ├── verify_env.py                 CUDA·패키지·프로젝트 모듈 임포트 확인
    ├── verify_model.py               배포 모델이 제대로 로드되는지 확인
    ├── smoke_catvton.py              데이터셋 없이 합성만 돌려보는 최소 검증
    └── scan_musinsa_categories.py    유효한 무신사 카테고리 코드 탐색
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

## 아직 남은 것 — 상품 사진 (학습이 아니라 합성 재료)

모델은 완성돼 있다. 빠진 것은 **합성에 넣을 상품 사진과 그 카탈로그**다.

`src/catvton_tryon.py:866` 이 `product.image_path` 로 옷 사진 파일을 찾는데,
저장소의 `data/products.csv`(80개)에는 **이미지 칼럼이 아예 없다**. 색상·소재·핏 같은
텍스트 속성만 있는 수작업 카탈로그라, 추천은 나와도 입힐 옷 사진이 없다.

팀원의 `scripts/musinsa_crawler.py` 가 이 구멍을 메운다 — 무신사 상품 사진을
`datasets/garments/raw/` 에 받고, `image_url`·`image_path` 칼럼이 들어간
`data/products_musinsa.csv` 를 따로 쓴다.

### 팀원에게 받을 것

| 경로 | 무엇 | 필수 |
| --- | --- | --- |
| `ai_fashion_recommender/data/products_musinsa.csv` | 이미지 칼럼이 있는 상품 카탈로그 | **필수** |
| `datasets/garments/raw/` | 상품 대표 사진 | **필수** |
| `datasets/people/{men,women}/` | 테스트용 인물 사진 | 선택 (없으면 `data/input_person.jpg` 사용) |

`datasets/garments/clean/` 은 파서가 자동 생성·캐시하므로 받을 필요 없다.

### 받은 뒤

`test_tryon.py:72` 가 `products_musinsa.csv` 가 있으면 자동으로 그걸 쓰므로
**설정을 바꿀 필요 없이 파일만 넣으면 된다.**

```bash
scp -P 37220 -r datasets dsl01@165.132.80.36:/data1/dsl01/26-2_Modeling_Fashion/
scp -P 37220 ai_fashion_recommender/data/products_musinsa.csv   dsl01@165.132.80.36:/data1/dsl01/26-2_Modeling_Fashion/ai_fashion_recommender/data/
```

### 알아둘 것

`web/pipeline.py:117` 은 `products.csv` 를 하드코딩한다. 즉 크롤링 카탈로그를 넣어도
**웹 앱은 여전히 사진 없는 80개 카탈로그를 쓴다.** 웹에서 합성까지 보여주려면
이 줄도 `test_tryon.py` 처럼 자동 선택으로 바꿔야 한다.

## 배포 모델

서버의 `fashion_attribute_heads.pt`는 저장소본, 즉 **옛 baseline(4,789 샘플)**이다.
내 PC의 augmented(22,341 crop)와 결과를 맞추려면 덮어쓴다.

```bash
scp -P 37220 ai_fashion_recommender/models/fashion_attribute_heads.pt \
  dsl01@165.132.80.36:/data1/dsl01/26-2_Modeling_Fashion/ai_fashion_recommender/models/
```

## 라이선스

CatVTON 가중치는 **CC BY-NC-SA 4.0(비상업)**. 발표·배포 범위를 정할 때 확인할 것.
