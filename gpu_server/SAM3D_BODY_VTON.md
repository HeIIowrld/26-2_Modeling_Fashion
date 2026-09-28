# SAM 3D Body 기반 넓은 옷 → 좁은 옷 VTON

작성: 2026-09-29

## 목적과 구조

기존 관절 capsule body proxy는 팔·다리의 중심은 알 수 있지만, 와이드 팬츠나
부피가 큰 아우터 아래 신체 폭을 직접 복원하지 못한다. 이번 구현은 Meta의
SAM 3D Body를 선택적 1차 geometry backend로 사용한다.

1. FITTA가 이미 찾은 주인공 사람 마스크를 SAM 3D Body의 bbox·mask prompt로 준다.
2. 단일 사진에서 복원한 MHR mesh를 원본 카메라로 투영해 옷에 가려진 신체 위치
   prior를 만든다. 이것은 실제 치수가 아니라 편집 영역용 확률적 추정이다.
3. 넓은 기존 옷에서 좁은 목표 옷으로 바뀔 때, `기존 옷 - 목표 몸선`의 바깥 고리를
   FLUX 인페인팅으로 먼저 배경 복원한다.
4. 복원된 사진의 몸선 주변만 CatVTON으로 새 상품을 입힌다.
5. 모델 파일이 없거나 복원 신뢰도가 낮으면 기존 관절 기반 proxy와 1-pass 전환으로
   자동 복귀한다. 웹 분석 자체를 실패시키지 않는다.

모델은 같은 GPU의 CatVTON·FLUX와 동시에 상주시킬 수 없다. 새 사진을 처리할 때
기존 생성 pipeline을 해제하고 SAM 3D Body를 실행한 뒤 mesh 투영 마스크만 최대
4개 캐시한다. 따라서 첫 합성 지연은 늘지만 OOM보다 결과 안정성을 우선한다.

## 서버 설치

SAM 3D Body 공식 저장소와 체크포인트는 Git에 넣지 않는다. 체크포인트는 Hugging
Face에서 접근 승인이 필요하다. 먼저 `facebook/sam-3d-body-dinov3` 사용 승인을
받고 master node에서 인증한다. 계산 노드는 인터넷이 없으므로 다운로드는 반드시
master node에서 끝낸다.

권장 배치 경로(master 기준):

```text
/data1/dsl01/releases/fitta_current/third_party/sam-3d-body/
/data1/dsl01/shared_models/sam-3d-body-dinov3/model.ckpt
/data1/dsl01/shared_models/sam-3d-body-dinov3/assets/mhr_model.pt
```

계산 노드에서는 `/data1` 대신 `/mnt/data1`이다. 공식 설치 절차는 Python 3.11,
PyTorch, detectron2 및 SAM 3D Body 의존성을 요구한다. 기존 운영 환경에 바로 덮지
말고 별도 평가 환경에서 import와 1장 추론을 먼저 확인한다.

```bash
cd /data1/dsl01/releases/fitta_current/third_party
git clone https://github.com/facebookresearch/sam-3d-body.git

# 접근 승인 후 master node에서 실행. 토큰은 셸 입력/사용자 캐시에만 두고
# 저장소 .env, 서비스 파일, 문서에 기록하지 않는다.
huggingface-cli login
huggingface-cli download facebook/sam-3d-body-dinov3 \
  --local-dir /data1/dsl01/shared_models/sam-3d-body-dinov3
```

공식 체크포인트 내부 실제 파일명이 다르면 아래 환경변수의 경로만 맞춘다.

```bash
FASHION_SAM3D_BODY_REPO=/mnt/data1/dsl01/releases/fitta_current/third_party/sam-3d-body
FASHION_SAM3D_BODY_CHECKPOINT=/mnt/data1/dsl01/shared_models/sam-3d-body-dinov3/model.ckpt
FASHION_SAM3D_BODY_MHR=/mnt/data1/dsl01/shared_models/sam-3d-body-dinov3/assets/mhr_model.pt
```

이 값은 사용자 systemd drop-in 또는 Slurm 제출 환경에 넣는다. 세 파일이 모두
확인된 경우에만 웹 factory가 새 backend를 켠다.

## 검증 순서

1. 와이드 팬츠 사진 + 스트레이트/슬림 상품, 볼륨 아우터 사진 + 레귤러 상의를
   각각 5장 이상 준비한다. 사용자 사진과 무신사 이미지는 Git에 올리지 않는다.
2. 로그/진단의 `body-mesh:sam-3d-body`와
   `sam3d-two-stage(background+catvton)` 발동을 확인한다.
   `/api/health`의 `body_geometry_backend`도 `sam-3d-body`여야 한다.
3. 기존 옷 외곽이 남는지, 배경에 구멍·신체 추가 생성이 생기는지, 얼굴·손·가방이
   유지되는지 원본/기존 proxy/새 방식 3열로 눈으로 비교한다.
4. 실패율, 1장 전체 시간, peak VRAM을 기록한다. mesh 신뢰도 거절 시 fallback도
   실제로 동작해야 한다.
5. 새 방식이 대표 평가셋에서 기존 방식보다 낫지 않으면 환경변수를 제거해 즉시
   기존 proxy로 되돌린다.

현재 로컬 자동 테스트는 mask prompt 전달, mesh 투영 겹침 검사, cache, 실패 fallback,
기존 옷 residual ring, background 복원 → CatVTON 2단계 연결을 검사한다. 실제 Meta
checkpoint 생성 품질과 시간은 GPU 서버에서 아직 측정해야 한다.

## 한계

- 단일 사진 신체 복원은 추정이며 실제 신체 치수나 신체 사실을 제공하지 않는다.
- 극단적 가림, 옆모습, 잘린 발, 여러 사람이 겹친 사진에서는 fallback될 수 있다.
- 배경 복원과 VTON을 두 번 생성하므로 기존 합성보다 느리다.
- SAM 3D Body, CatVTON, FLUX 각각의 라이선스를 배포 전에 별도로 확인해야 한다.
- 입력 사진과 추론 mesh/mask의 저장 기간·접근 권한을 운영 개인정보 정책에 맞춰야 한다.
