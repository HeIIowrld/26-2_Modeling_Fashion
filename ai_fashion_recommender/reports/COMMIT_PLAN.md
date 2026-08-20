# 커밋 계획 (미실행 — 저장소 준비 후 사용)

> 상태: **커밋하지 않음.** 이 프로젝트 폴더에는 Git 저장소가 없습니다
> (`.git` 없음, `origin` 없음, 상위 4단계까지 확인).
> 아래는 저장소가 준비됐을 때 그대로 쓸 수 있는 목록입니다.
> 사전 안전 점검은 통과했습니다 — `reports/20_preflight_commit_check.json`

## 요약

| 항목 | 값 |
|---|---|
| 커밋 예정 | **77개 / 11.30 MB** |
| 100MB 이상 파일 | 0 |
| 개인정보(`<user-b>`·이메일·임시경로) | **0건** (전수 검색) |
| manifest CSV 절대경로 | **0건** |
| 배포 모델 | `models/fashion_attribute_heads_augmented.pt` (13.06 MB) — 변경 없음 |

## 커밋 대상

| 그룹 | 개수 | 용량 |
|---|---|---|
| `reports/` (md·json·manifests csv) | 33 | 6.08 MB |
| `experiments/` (py) | 30 | 0.25 MB |
| `data/` (manifest·selection·csv·shard_index) | 11 | 4.95 MB |
| `models/fashion_attribute_heads_augmented_r3.metrics.json` | 1 | 0.01 MB |
| `.gitignore` | 2 | — |

## 커밋 제외 (`.gitignore` 적용)

| 대상 | 용량 | 이유 |
|---|---|---|
| `models/fashion_attribute_heads_augmented_r3*.pt` (4개) | 54.6 MB | 3차 미채택 실험물 |
| `models/fashion_attribute_heads_detailnone.pt` | 13.68 MB | 미채택 중간 실험물 |
| `models/fashion_attribute_heads_finetuned.pt` | 13.70 MB | 미채택 중간 실험물 |
| `data/**/images/` | 약 1.2 GB | 원본 이미지, 라이선스·용량 |
| `data/**/instances_subset.json` | 16.8 MB | 원본 주석 파생본 |
| `data/provenance/**/*train2020*.json`, `*val2020*.json` | 557 MB | 원본 주석 |
| `data/cache/**` | 약 135 MB | 임베딩 캐시 (재생성 가능) |
| `reports/*.log` | — | 임시 로그 |

## 판단 보류 — `models/*.pt` 2개

| 파일 | 크기 | 처리 |
|---|---|---|
| `fashion_attribute_heads_augmented.pt` | 13.06 MB | **배포 모델.** 저장소가 생기면 이 파일 하나는 포함 권장 (100MB 제한 여유) |
| `fashion_attribute_heads.pt` | 13.67 MB | **현재 untracked → 이번에 새로 추가하지 않음.** 기존 저장소에서 이미 추적 중이면 그대로 유지 |

두 파일 모두 `.gitignore`에 넣지 **않았습니다.** 기존 저장소에서 이미 추적 중일 경우
무시 규칙이 추적 상태와 충돌하지 않도록 하기 위해서입니다.

## 저장소가 준비되면

```bash
cd ai_fashion_recommender

# 1) 기존 clone 이 따로 있으면 그 경로에서 작업할 것
git switch -c feature/fashion-attributes-round3

# 2) 이번 작업 파일만 명시적으로 stage
git add .gitignore ../.gitignore
git add experiments/
git add reports/
git add data/fashionpedia_train/manifest.json \
        data/fashionpedia_train/manifest_r2.json \
        data/fashionpedia_train/manifest_r2_corrected.json \
        data/fashionpedia_train/selection.json \
        data/fashionpedia_train/selection_r2.json \
        data/fashionpedia_train/fashion_attribute_annotations.csv \
        data/fashionpedia_train/fashion_attribute_annotations_r2.csv
git add data/fashionpedia_train_r3/manifest.json \
        data/fashionpedia_train_r3/selection_r3.json \
        data/fashionpedia_train_r3/fashion_attribute_annotations.csv
git add data/provenance/fashionpedia/shard_index.json
git add models/fashion_attribute_heads_augmented_r3.metrics.json
# 배포 모델을 함께 올릴 경우에만:
# git add models/fashion_attribute_heads_augmented.pt

# 3) stage 결과 확인 (커밋 전 필수)
git status --short
git diff --cached --stat
git ls-files -s --cached | wc -l

# 4) 커밋 — force push 금지
git commit -m "feat: add round 3 fashion attribute augmentation pipeline"
git push -u origin feature/fashion-attributes-round3
```

`reports/manifests/README.md`는 커밋 후에도 그대로 유효합니다.
4차 후보 선정 시에는 1·2·3차 선택분 **16,500개**를 제외해야 합니다
(`fashionpedia_train_excluded_image_ids.json` 10,500 + `fashionpedia_train_r3_images.csv` 6,000).

## 사전 점검에서 남은 경고 4건 — 전부 오탐

| 파일 | 내용 |
|---|---|
| `experiments/preflight_commit_check.py` | 검사기 자신의 정규식 리터럴 |
| `experiments/verify_provenance_manifests.py` | 같음 |
| `reports/20_preflight_commit_check.json` | 위 검사 결과를 기록한 자기참조 |
| `reports/FINAL_REPORT.md` | 산문 속 "OneDrive" 단어 (경로 아님) |

실제 누출이던 2건(`reports/01_baseline_diagnosis.json`, `reports/08_relabel_summary.json`의
사용자명 포함 절대경로)은 저장소 상대경로로 치환해 해소했습니다.
