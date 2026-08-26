"""학습 속성 헤드가 없거나 보류될 때 사용하는 FashionSigLIP zero-shot 후보군."""

STYLE_PROMPTS = {
    "캐주얼": "a casual everyday outfit",
    "미니멀": "a clean minimal outfit",
    "포멀": "a formal business outfit",
    "스트리트": "a streetwear outfit",
    "로맨틱": "a romantic outfit",
    "스포티": "a sporty outfit",
}

UPPER_TYPE_PROMPTS = {
    "티셔츠": "a basic collarless t-shirt",
    "폴로 셔츠": "a polo shirt with a ribbed polo collar and placket",
    "셔츠": "a button-up shirt",
    "블라우스": "a blouse",
    "니트": "a knit sweater",
    "가디건": "a button-front knitted cardigan",
    "후드티": "a hoodie",
    "재킷": "a casual jacket",
    "블레이저": "a structured tailored blazer",
    "코트": "a long outerwear coat",
    "베스트": "a sleeveless vest worn as an outer layer",
    "탑": "a fitted sleeveless fashion top",
    "원피스": "a one-piece dress",
    "점프수트": "a one-piece jumpsuit with trouser legs",
}

LOWER_TYPE_PROMPTS = {
    "청바지": "a pair of denim jeans",
    "슬랙스·팬츠": "a pair of tailored trousers or pants",
    "조거팬츠": "a pair of jogger pants",
    "반바지": "a pair of shorts",
    "스커트": "a skirt",
}

PATTERN_PROMPTS = {
    "플로럴": "a garment with a floral pattern",
    "그래픽": "a garment with a graphic print or logo",
    "스트라이프": "a striped garment",
    "무지": "a solid pure color garment with no pattern",
    "체크": "a checkered plaid lattice garment",
    "컬러 블록": "a color block garment with distinct blocks of color",
    "기타 패턴": "a garment with another repeating pattern",
}

MATERIAL_PROMPTS = {
    "데님 추정": "a denim fabric garment",
    "코튼 추정": "a cotton fabric garment",
    "가죽 추정": "a leather garment",
    "퍼·플리스 추정": "a furry fleece garment",
    "니트 추정": "a knitted fabric garment",
    "시폰 추정": "a chiffon fabric garment",
    "기타 소재": "a garment made from another fabric",
}

NECKLINE_PROMPTS = {
    "V넥": "a garment with a V shaped neckline",
    "스퀘어넥": "a garment with a square neckline",
    "라운드넥": "a garment with a round crew neckline",
    "스탠드 칼라": "a garment with a standing collar",
    "라펠 칼라": "a garment with a lapel collar",
    "서스펜더·슬링": "a suspender or sling garment neckline",
}

# 현재 배포 체크포인트에는 layering_state 헤드가 없다. 정식 헤드를 학습하기 전까지
# 상의 전체와 목 ROI에서 사용하는 보조적인 zero-shot 후보군이다.
LAYERING_PROMPTS = {
    "단일 상의": "one single upper body garment, not layered with another top",
    "레이어드": "a visibly layered outfit with a shirt worn underneath a knit sweater or vest",
}
