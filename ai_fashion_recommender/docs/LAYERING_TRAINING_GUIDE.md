# 레이어드 헤드 학습 준비

기존 `fashion_attribute_heads_augmented.pt`는 그대로 사용한다. 새 데이터는
`data/layering_labels_template.csv` 형식으로 정리하고 별도
`models/layering_heads.pt`를 만든다.

필수 열:

- `image_path`: 이미지 경로
- `split`: `train` 또는 `val`
- `is_layered`: `0/1` 또는 `단일 옷/겹쳐입음`

선택 열:

- `bbox_x,bbox_y,bbox_w,bbox_h`: 전신사진에서 상의 영역만 자를 때 사용
- `inner_category`: 겹쳐입은 경우 안쪽 옷
- `outer_category`: 겹쳐입은 경우 바깥쪽 옷

학습 명령:

```bash
python scripts/train_layering_heads.py \
  --csv data/layering_labels.csv \
  --image-root data/layering_images \
  --device cuda \
  --rebuild-cache
```

각 이미지에서 전체 상의, 목·칼라, 좌우 커프스, 밑단, 앞여밈 ROI를 추출해
FashionSigLIP 임베딩을 만들고 세 헤드를 함께 학습한다.

- `layering`: 단일 옷 / 겹쳐입음
- `inner_category`: 안쪽 옷 종류
- `outer_category`: 바깥쪽 옷 종류

레이어드 확률 0.30~0.70은 추론 시 `판단 보류`로 처리한다. 체크포인트가 아직
없으면 기존 zero-shot 및 속성 충돌 규칙을 자동으로 사용한다.
