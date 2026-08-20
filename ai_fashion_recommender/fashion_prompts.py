"""DeepFashion-MultiModal 라벨에 맞춘 FashionSigLIP zero-shot 후보군."""

STYLE_PROMPTS = {
    "캐주얼": "a casual everyday outfit",
    "미니멀": "a clean minimal outfit",
    "포멀": "a formal business outfit",
    "스트리트": "a streetwear outfit",
    "로맨틱": "a romantic outfit",
    "스포티": "a sporty outfit",
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
