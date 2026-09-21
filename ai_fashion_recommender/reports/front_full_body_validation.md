# 정면 전신사진 입력 유도 및 검증

## 목적

추천 분석 전에 사용자가 정면 전신사진을 올리도록 안내하고, MediaPipe landmark가 충분히 신뢰되는 경우에만 명확한 머리·발 프레이밍, 비정면 자세, 팔 자세 위반을 재촬영 대상으로 판정한다. visibility가 낮거나 신호가 서로 충돌하면 `uncertain`으로 남겨 hard rejection을 피한다.

사진 자체를 확인할 수 없는 경우는 예외다. 사람이 감지되지 않거나 pose가 유효하지 않으면 hard failure로 처리하며, 양쪽 어깨와 골반 등 핵심 torso landmark가 없거나 비어 있거나 모두 NaN이어도 hard failure로 처리한다. 사람은 감지되었지만 발·손목 등 일부 landmark visibility만 낮은 경우에는 기존처럼 `uncertain`을 유지한다.

사진 선택 직후에는 미리보기만 표시한다. 사용자가 사진 단계에서 다음을 누르면 `POST /api/validate-photo`가 기존 `PoseAnalyzer`와 `QualityChecker`로 preflight 검증을 수행하고, 부적합하면 조건 입력 단계로 이동하지 않고 업로드 영역에 재촬영 메시지를 표시한다. 적합한 사진만 조건 입력 단계로 이동한다. 전체 `/api/analyze` 경로에서도 동일한 검증을 다시 수행해 preflight를 우회한 요청을 차단한다.

## UI 입력 가이드

첫 화면 `STEP 01` 업로드 영역 옆에 `web/static/assets/full-body-guide.png`를 표시한다.

- 정면 전신사진을 올려주세요
- 머리부터 발끝까지 모두 나오게 촬영
- 몸과 얼굴은 정면을 향하기
- 팔은 몸 옆에 자연스럽게 내리기
- 몸을 가리는 물건 없이 한 명만 촬영

## 판정 상태

### Framing

| 항목 | 상태 | hard failure 조건 |
|---|---|---|
| `head_framing` | `visible`, `cropped`, `uncertain` | reliable nose의 `y <= 0.03` |
| `lower_body_framing` | `visible`, `cropped`, `uncertain` | 기존처럼 reliable ankle/foot 좌표가 frame 밖 |
| `full_body_framing` | 세 상태의 결합 | head 또는 lower가 `cropped` |

머리끝은 nose 하나로 직접 검출하지 않는다. nose visibility가 `0.5` 미만이면 `uncertain`이고, 신뢰되는 nose가 상단 3% 안쪽일 때만 명확한 상단 crop으로 본다. 발 검사는 팀장 변경사항의 기준을 그대로 유지한다.

### 정면 자세

사용한 derived metric:

- 양쪽 shoulder/hip visibility threshold: `0.7`
- shoulder line tilt: 기존 `0.06`
- shoulder midpoint와 hip midpoint의 x 차이: 기존 `0.08`
- 좌우 landmark visibility 차이: 최대 `0.25`
- shoulder tilt 명확한 위반: `0.14`, severe: `0.22`
- torso offset 명확한 위반: `0.18`, severe: `0.28`
- shoulder width / torso height 허용 범위: `0.35` ~ `2.5`

두 값 중 하나만 약간 벗어난 경우에는 `uncertain`으로 둔다. 두 지표가 함께 명확히 벗어나거나 severe 신호와 비정상적인 상대 비율이 동시에 있을 때만 `non_front`로 판정한다.

### 팔 자세

사용 landmark: 양쪽 shoulder, elbow, wrist, hip.

- visibility threshold: `0.7`
- 손목 중심 교차 margin: `0.08`
- 어깨보다 위로 올라간 손목 margin: `0.08`
- 어깨 바깥쪽 open margin: shoulder width의 `0.25`
- 같은 쪽 어깨에서 손목까지 허용 거리: shoulder width의 `1.5`

`arms_down`은 양쪽 손목이 hip 부근 또는 아래에 있고, 팔꿈치가 shoulder와 wrist 사이 높이에 있으며, 손목이 몸통 중심을 가로지르지 않을 때다. visibility가 낮거나 조건이 애매하면 `uncertain`이다. 명확한 raised/open 또는 crossed만 hard failure로 처리한다.

## 사용자 오류 안내

실패 메시지는 최대 3개로 제한한다.

- 머리/발 framing: 머리부터 발끝까지 모두 나오게 전신사진으로 다시 촬영
- 비정면: 몸을 정면으로 향하고 양쪽 어깨가 모두 보이도록 다시 촬영
- 팔 위반: 팔을 몸 옆에 자연스럽게 내리고 몸통을 가리지 않도록 다시 촬영

기존 `/api/jobs/{job_id}`의 `error`와 웹 error card 전달 구조는 유지한다.

사람을 확인할 수 없는 경우에는 다음 취지의 메시지를 사용한다.

- 사진에서 사람의 정면 전신을 확인할 수 없습니다. 머리부터 발끝까지 한 명만 나오도록 다시 촬영해 주세요.

preflight의 정상 전신사진 통과, 부적합 사진 차단, 검증 중 상태 표시, 업로드 영역 오류 표시, 새 사진 선택 시 오류 초기화는 웹 테스트에서 확인한다.

## 한계

- nose는 머리카락이나 실제 머리끝을 직접 나타내지 않으므로 보수적인 상단 crop 신호로만 사용한다.
- MediaPipe landmark만으로 카메라 방향과 실제 신체 방향을 완전히 분리할 수 없다.
- 팔 자세는 self-occlusion과 옷 형태의 영향을 받는다.
- threshold는 landmark mock과 기존 단일 입력 기준으로 시작한 값이며, 다양한 정면 전신사진 validation set으로 재보정해야 한다.
