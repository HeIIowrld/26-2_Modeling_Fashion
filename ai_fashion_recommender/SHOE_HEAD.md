# 신발 인식 헤드 (학습 전)

`src/shoe_model.py`는 기존 의류 속성 헤드와 **별도 체크포인트**를 사용한다.
기존 `ATTRIBUTE_TASKS`와 의류 가중치는 변경하지 않았다. 아직 신발 가중치나 정확도 결과는 없다.

## 데이터와 학습 계약

1. 전신 사진에서 FASHN의 feet(15) bbox를 `shoe_crop`으로 추출한다. feet는 맨발도 포함하므로 신발 검출 정답 자체가 아니다.
2. crop에 `SHOE_LABELS`의 종류를 라벨링한다. 맨발·기타 신발도 포함하고 발이 잘리거나 보이지 않는 표본은 분리한다.
3. 사람/촬영 세션/상품이 train·validation·test 사이에 중복되지 않게 분리한다.
4. 기존 FashionSigLIP의 squash 전처리와 정규화 이미지 임베딩을 사용한다. 백본은 고정하고 헤드만 학습한다.
5. 클래스별 precision/recall, macro-F1, 맨발 오검출, 신뢰도별 보류 비율을 검증하고 confidence 임계값을 정한다.

라벨: 스니커즈, 러닝화, 로퍼, 더비슈즈, 메리제인, 펌프스, 샌들, 슬리퍼, 부츠, 워커, 기타 신발, 맨발.
상품 단독 사진만으로 학습하면 착장 사진과 차이가 크므로 실제 착장 crop을 반드시 포함한다.
현재 bbox는 양발을 함께 포함한다. 학습 때도 동일한 crop 정의를 사용해야 한다.

## 학습 코드에서 사용할 인터페이스

```python
import torch
from shoe_model import build_shoe_heads, shoe_classification_loss, save_shoe_checkpoint

# features: [batch, embedding_dim], labels: SHOE_LABELS 인덱스(-1은 미라벨)
heads = build_shoe_heads(input_dim=features.shape[-1]).to(features.device)
optimizer = torch.optim.AdamW(heads.parameters(), lr=1e-3)
heads.train()
optimizer.zero_grad()
loss = shoe_classification_loss(heads(features.detach()), labels)
loss.backward()
optimizer.step()

# 실제 전체 학습과 별도 검증이 끝난 후에만 저장한다.
save_shoe_checkpoint(
    "models/shoe_heads.pt", heads, backbone_model_id=classifier.model_id,
    training_examples=len(train_dataset), threshold=validated_threshold,
)
```

위는 인터페이스 예시이며 데이터 로더/학습 루프/검증기를 대신하지 않는다. 데이터 제공 후 연결한다.
`save_shoe_checkpoint`의 학습 표본 수는 호출자가 기록하는 메타데이터이며, 학습 품질을 인증하지 않는다.

## 추론 활성화

학습된 파일을 만든 뒤 서버/노트북 실행 전에 macOS 터미널에서:

```bash
export FASHION_SHOE_CHECKPOINT="/절대경로/models/shoe_heads.pt"
```

FashionSigLIP이 켜져 있어야 하며, 학습 때와 동일한 백본 ID가 필요하다.
환경변수/가중치가 없으면 `not_trained`, 발 영역이 없으면 `not_visible`, 낮은 신뢰도·맨발·기타 신발이면 `uncertain`이다.
인식값은 `OutfitAnalysis.shoes`와 웹의 내 착장 분석에 표시된다.
명시 목적/스타일이 없는 경우에만 인식된 신발 종류를 추천 키워드의 보충값으로 사용한다.
신발은 체형 적합도 점수와 CatVTON 합성 대상에 추가하지 않았다.

## 이후 트렌드 반영안

날짜·시즌·출처가 있는 허용된 영상 자막/룩북 → 코디 조합·상황·신발 특징 구조화 → 사람 검수 →
최근성·여러 출처 합의에 가중치를 둔 트렌드 저장소 → 규칙으로 후보 생성 후 트렌드로 재정렬 →
사용자 선호 쌍 비교/클릭/저장으로 검증·보정.

Fashion Rules는 필수 조건/안전장치로, 트렌드는 교체 가능한 선호 신호로 분리한다.
영상 조회수만 취향의 정답으로 삼지 않고 광고·특정 채널 편향을 분리한다. 현재 이 트렌드 수집/재정렬은 구현 범위가 아니다.
