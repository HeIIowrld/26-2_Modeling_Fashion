from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AttributeTask:
    name: str
    labels: tuple[str, ...]
    multi_label: bool = False
    minimum_confidence: float = 0.50


# 빈 라벨은 "없음"이 아니라 미주석으로 처리한다. 실제 부재는 무지·칼라 없음처럼
# 명시적인 클래스로 기록해야 학습 시 "안 보임"과 "없음"이 섞이지 않는다.
ATTRIBUTE_TASKS = {
    "category": AttributeTask(
        "category",
        (
            "티셔츠", "폴로 셔츠", "셔츠", "블라우스", "니트", "가디건", "후드티", "재킷",
            "블레이저", "코트", "베스트", "탑", "팬츠", "청바지", "쇼츠",
            "스커트", "원피스", "점프수트",
        ),
        minimum_confidence=0.45,
    ),
    "lower_subtype": AttributeTask(
        "lower_subtype",
        (
            "슬랙스", "치노 팬츠", "청바지", "카고 팬츠", "조거·스웨트팬츠",
            "트랙팬츠", "레깅스", "하렘 팬츠", "요가 팬츠", "세일러 팬츠",
        ),
        minimum_confidence=0.48,
    ),
    # 바지 이름과 다리 모양은 서로 독립적이다. 예: 카고 팬츠이면서 와이드일 수 있다.
    "pant_leg_shape": AttributeTask(
        "pant_leg_shape",
        (
            "스키니", "스트레이트", "테이퍼드", "페그", "부츠컷", "플레어",
            "와이드", "팔라초",
        ),
        minimum_confidence=0.45,
    ),
    "pant_length": AttributeTask(
        "pant_length", ("카프리·7부", "크롭·앵클", "풀렝스"), minimum_confidence=0.48
    ),
    "lower_detail": AttributeTask(
        "lower_detail",
        (
            "5포켓", "카고 포켓", "플리츠·턱", "드로스트링", "밴딩 허리",
            "사이드 스트라이프", "밑단 커프", "스터럽",
        ),
        multi_label=True,
        minimum_confidence=0.45,
    ),
    "sleeve_length": AttributeTask(
        "sleeve_length", ("민소매", "반팔", "7부 소매", "긴팔"), minimum_confidence=0.50
    ),
    "sleeve_shape": AttributeTask(
        "sleeve_shape",
        ("기본 소매", "퍼프 소매", "벌룬 소매", "벨 소매", "래글런 소매", "캡 소매"),
        minimum_confidence=0.50,
    ),
    "upper_length": AttributeTask(
        "upper_length", ("크롭 기장", "기본 기장", "롱 기장"), minimum_confidence=0.50
    ),
    "lower_length": AttributeTask(
        "lower_length",
        ("쇼츠·미니 기장", "무릎 기장", "미디·7부 기장", "롱·긴바지 기장"),
        minimum_confidence=0.50,
    ),
    "neckline": AttributeTask(
        "neckline",
        ("라운드넥", "V넥", "스퀘어넥", "보트넥", "오프숄더", "홀터넥", "터틀넥"),
        minimum_confidence=0.52,
    ),
    "collar": AttributeTask(
        "collar",
        ("칼라 없음", "폴로 칼라", "셔츠 칼라", "스탠드 칼라", "라펠 칼라", "피터팬 칼라", "세일러 칼라"),
        minimum_confidence=0.52,
    ),
    "upper_fit": AttributeTask(
        "upper_fit", ("슬림핏", "레귤러핏", "여유핏", "오버핏"), minimum_confidence=0.50
    ),
    "lower_fit": AttributeTask(
        "lower_fit",
        ("슬림핏", "스트레이트핏", "테이퍼드핏", "와이드핏", "플레어핏"),
        minimum_confidence=0.50,
    ),
    "silhouette": AttributeTask(
        "silhouette", ("H라인", "A라인", "X라인", "O라인", "비대칭"), minimum_confidence=0.50
    ),
    "pattern": AttributeTask(
        "pattern",
        ("무지", "스트라이프", "체크", "도트", "플로럴", "그래픽", "애니멀", "카모", "컬러 블록"),
        multi_label=True,
        minimum_confidence=0.45,
    ),
    "material": AttributeTask(
        "material",
        ("코튼", "데님", "니트", "울", "가죽", "스웨이드", "시폰", "실크·새틴", "퍼·플리스", "메시", "레이스"),
        multi_label=True,
        minimum_confidence=0.48,
    ),
    "detail": AttributeTask(
        "detail",
        ("디테일 없음", "포켓", "프릴·러플", "지퍼", "벨트", "단추", "리본", "레이스", "플리츠", "자수", "스팽글", "후드"),
        multi_label=True,
        minimum_confidence=0.45,
    ),
}


UPPER_CATEGORIES = {
    "티셔츠", "폴로 셔츠", "셔츠", "블라우스", "니트", "가디건", "후드티", "재킷",
    "블레이저", "코트", "베스트", "탑", "원피스", "점프수트",
}
LOWER_CATEGORIES = {"팬츠", "청바지", "쇼츠", "스커트"}
UPPER_ONLY_TASKS = {"sleeve_length", "sleeve_shape", "upper_length", "neckline", "collar", "upper_fit"}
LOWER_ONLY_TASKS = {
    "lower_subtype", "pant_leg_shape", "pant_length", "lower_detail", "lower_length", "lower_fit",
}
MULTI_LABEL_TASKS = {name for name, task in ATTRIBUTE_TASKS.items() if task.multi_label}
SINGLE_LABEL_TASKS = set(ATTRIBUTE_TASKS) - MULTI_LABEL_TASKS


def label_index(task_name: str) -> dict[str, int]:
    return {label: index for index, label in enumerate(ATTRIBUTE_TASKS[task_name].labels)}


def validate_attribute_schema() -> None:
    for name, task in ATTRIBUTE_TASKS.items():
        if not task.labels:
            raise ValueError(f"{name} 속성에 라벨이 없습니다.")
        if len(task.labels) != len(set(task.labels)):
            raise ValueError(f"{name} 속성에 중복 라벨이 있습니다.")


validate_attribute_schema()
