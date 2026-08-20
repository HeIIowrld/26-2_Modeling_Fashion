from __future__ import annotations

"""무신사 상품 리스트 수집기.

무신사 웹사이트가 내부적으로 사용하는 상품 리스트 JSON API를 호출해
카테고리별 인기 상품의 메타데이터와 전면 썸네일을 수집한다.

주의사항
- 연구·학습용 소규모 수집 전용이다. 요청 간격을 지키고 수백 개 수준만 수집한다.
- 이미지는 로컬 캐시(data/musinsa_images/)에만 저장하고 재배포하지 않는다.
- 결과 CSV에는 원본 상품 URL을 함께 기록해 출처를 유지한다.

사용법:
    python musinsa_crawler.py --per-category 60 --delay 1.0
"""

import argparse
import csv
import json
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import sys

# 런타임 모듈은 src/에 있다. 임포트 전에 경로를 등록한다.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import DATA_DIR, GARMENT_RAW_DIR
from schemas import BODY_SHAPES

API_URL = "https://api.musinsa.com/api2/dp/v2/plp/goods"
# size=100이 1페이지에서 허용하는 최대치다(그 이상은 400 Bad Request).
PAGE_SIZE = 100
# 2페이지 이상은 프론트엔드가 계산하는 서명(hmacId)이 있어야 접근 가능해 그대로
# 요청하면 매번 403으로 막힌다. 서명을 역산하는 대신 요청당 최대치인 size=100으로
# 1페이지만 받아 불필요한 403 재시도 대기를 없앤다.
MAX_PAGES = 1

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Referer": "https://www.musinsa.com/",
    "Accept-Language": "ko-KR,ko;q=0.9",
}

# 무신사 카테고리 코드 → 프로젝트 카테고리(top/bottom)
# 실제 API를 호출해 확인한 코드다(2026-08 기준, 무신사 개편 시 바뀔 수 있음):
# 001 상의, 002 아우터, 003 바지, 004 가방, 100 스커트, 103 신발.
# 추천 엔진(recommendation_engine.py)과 VTON 마스크 매핑(catvton_tryon.py)은 아직
# top/bottom 2종만 지원하므로, 우선 상/하의 볼륨을 늘리는 데 집중한다.
CATEGORY_MAP = {
    "001": "top",     # 상의
    "003": "bottom",  # 바지
    "100": "bottom",  # 스커트 (하의로 취급)
}

# 인기순 상위만 반복 수집하면 매번 같은 상품만 쌓인다. 정렬 기준을 섞어 색상·스타일
# 다양성을 넓힌다.
SORT_CODES = ["POPULAR", "NEW"]

# 무신사 이미지 CDN은 같은 파일을 여러 해상도로 제공한다(_500=500x600, _big=1500x1800).
# 합성용 원본 디테일(패턴·로고·재질)을 살리려면 큰 이미지를 받아야 한다.
IMAGE_SUFFIX_CANDIDATES = ["_big.jpg", "_500.jpg"]

# 상품명 키워드 → 추천 엔진이 사용하는 대표 색상명
COLOR_KEYWORDS = [
    ("블랙", ["블랙", "BLACK", "Black", "black", "챠콜", "차콜"]),
    ("화이트", ["화이트", "WHITE", "White", "white", "아이보리", "IVORY", "Ivory", "크림", "CREAM", "Cream"]),
    ("그레이", ["그레이", "GRAY", "GREY", "Gray", "Grey", "멜란지"]),
    ("네이비", ["네이비", "NAVY", "Navy", "navy"]),
    ("블루", ["블루", "BLUE", "Blue", "blue", "데님", "연청", "중청", "진청"]),
    ("브라운", ["브라운", "BROWN", "Brown", "brown", "카키", "KHAKI", "Khaki", "탄", "TAN"]),
    ("베이지", ["베이지", "BEIGE", "Beige", "beige", "샌드", "SAND", "오트밀"]),
    ("레드", ["레드", "RED", "Red", "버건디", "와인"]),
    ("핑크", ["핑크", "PINK", "Pink"]),
    ("그린", ["그린", "GREEN", "Green", "올리브", "OLIVE", "Olive"]),
    ("옐로", ["옐로", "YELLOW", "Yellow", "머스타드"]),
    ("퍼플", ["퍼플", "PURPLE", "Purple", "라벤더"]),
]

# 상품명 키워드 → 추천 엔진 스타일 분류
STYLE_KEYWORDS = [
    ("포멀", ["슬랙스", "테일러드", "수트", "셋업", "블레이저", "정장"]),
    ("스포티", ["조거", "트랙", "스웨트", "저지", "트레이닝", "윈드"]),
    ("스트리트", ["그래픽", "프린트", "오버사이즈", "빅로고", "카고"]),
    ("로맨틱", ["플레어", "리본", "레이스", "프릴"]),
    ("미니멀", ["베이직", "에센셜", "미니멀", "무지", "스탠다드", "클래식"]),
]

# 상품명 키워드 → 계절
SEASON_KEYWORDS = [
    ("여름", ["반소매", "반팔", "숏슬리브", "하프", "쿨", "린넨", "리넨", "숏츠", "버뮤다"]),
    ("겨울", ["기모", "울", "플리스", "헤비", "패딩", "니트"]),
]


@dataclass
class CrawledProduct:
    product_id: str
    name: str
    category: str
    color: str
    style: str
    purposes: str
    body_shapes: str
    price: int
    season: str
    stock: bool
    url: str
    brand: str = ""
    gender: str = ""
    image_url: str = ""
    image_path: str = ""


def _match_keyword(name: str, table: list[tuple[str, list[str]]], default: str) -> str:
    for label, keywords in table:
        if any(keyword in name for keyword in keywords):
            return label
    return default


def _guess_purposes(name: str, style: str) -> str:
    purposes = ["데일리"]
    if style == "포멀" or any(k in name for k in ("셔츠", "슬랙스", "블레이저")):
        purposes.append("출근")
    if style in ("미니멀", "로맨틱"):
        purposes.append("데이트")
    if style == "스포티":
        purposes.append("여행")
    return "|".join(dict.fromkeys(purposes))


def _open_with_retry(request: urllib.request.Request, retries: int = 3) -> bytes:
    """403/429/5xx는 일시적 차단일 수 있어 간격을 늘려가며 재시도한다."""
    wait = 10.0
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if attempt >= retries or exc.code not in (403, 429, 500, 502, 503):
                raise
            print(f"HTTP {exc.code} → {wait:.0f}초 대기 후 재시도 ({attempt + 1}/{retries})")
            time.sleep(wait)
            wait *= 3
    raise RuntimeError("unreachable")


def fetch_page(category_code: str, page: int, sort_code: str = "POPULAR", size: int = PAGE_SIZE) -> list[dict]:
    params = urllib.parse.urlencode({
        "gf": "A",
        "category": category_code,
        "sortCode": sort_code,
        "page": page,
        "size": size,
        "caller": "CATEGORY",
    })
    request = urllib.request.Request(f"{API_URL}?{params}", headers=HEADERS)
    payload = json.loads(_open_with_retry(request).decode("utf-8"))
    return payload.get("data", {}).get("list", [])


def parse_item(item: dict, category: str) -> CrawledProduct | None:
    goods_no = item.get("goodsNo")
    name = (item.get("goodsName") or "").strip()
    if not goods_no or not name or item.get("isSoldOut"):
        return None
    price = int(item.get("finalPrice") or item.get("price") or 0)
    if price <= 0:
        return None
    style = _match_keyword(name, STYLE_KEYWORDS, "캐주얼")
    return CrawledProduct(
        product_id=f"MS{goods_no}",
        name=name,
        category=category,
        color=_match_keyword(name, COLOR_KEYWORDS, "그레이"),
        style=style,
        purposes=_guess_purposes(name, style),
        # 상품 이미지만으로 체형 적합도를 알 수 없어 전 체형 허용으로 두고,
        # 이후 FashionSigLIP 속성 분석으로 세분화한다.
        body_shapes="|".join(BODY_SHAPES),
        price=price,
        season=_match_keyword(name, SEASON_KEYWORDS, "사계절"),
        stock=True,
        url=item.get("goodsLinkUrl") or f"https://www.musinsa.com/products/{goods_no}",
        brand=item.get("brandName") or item.get("brand") or "",
        gender=item.get("displayGenderText") or "공용",
        image_url=item.get("thumbnail") or "",
    )


def _image_url_candidates(thumbnail_url: str) -> list[str]:
    """썸네일 URL(보통 `..._500.jpg`)에서 더 큰 해상도 변형을 우선순위대로 만든다.

    무신사 이미지 CDN은 같은 파일을 `_big.jpg`(1500x1800 내외)로도 제공하는데,
    합성 시 옷의 패턴·로고·재질 디테일이 훨씬 잘 살아난다. 상품에 따라 `_big`이
    없을 수 있어 원래 `_500` 크기로 자동 대체한다.
    """
    url = thumbnail_url
    if url.startswith("//"):
        url = "https:" + url
    for suffix in IMAGE_SUFFIX_CANDIDATES:
        if url.endswith(suffix):
            base = url[: -len(suffix)]
            return [base + candidate for candidate in IMAGE_SUFFIX_CANDIDATES]
    return [url]


def download_image(product: CrawledProduct, image_dir: Path, delay: float, refresh: bool = False) -> None:
    if not product.image_url:
        return
    target = image_dir / f"{product.product_id}.jpg"
    if target.exists() and not refresh:
        product.image_path = str(target.relative_to(DATA_DIR.parent)).replace("\\", "/")
        return
    candidates = _image_url_candidates(product.image_url)
    last_error: Exception | None = None
    for index, url in enumerate(candidates):
        request = urllib.request.Request(url, headers=HEADERS)
        try:
            data = _open_with_retry(request, retries=1 if index < len(candidates) - 1 else 3)
        except Exception as exc:  # 큰 이미지가 없으면(404 등) 다음 후보로 넘어간다.
            last_error = exc
            continue
        target.write_bytes(data)
        product.image_path = str(target.relative_to(DATA_DIR.parent)).replace("\\", "/")
        time.sleep(delay)
        return
    raise last_error or RuntimeError(f"이미지 다운로드 실패: {product.image_url}")


def crawl(per_category: int, delay: float, max_price: int | None = None) -> list[CrawledProduct]:
    products: list[CrawledProduct] = []
    seen: set[str] = set()
    sort_quotas = _split_quota(per_category, len(SORT_CODES))
    for code, category in CATEGORY_MAP.items():
        category_collected = 0
        for sort_code, quota in zip(SORT_CODES, sort_quotas):
            collected = 0
            page = 1
            while collected < quota and page <= MAX_PAGES:
                try:
                    items = fetch_page(code, page, sort_code=sort_code)
                except Exception as exc:  # 한 카테고리 실패로 전체 수집을 잃지 않는다.
                    print(f"카테고리 {code}({sort_code}) {page}페이지 수집 중단: {exc}")
                    break
                if not items:
                    break
                for item in items:
                    product = parse_item(item, category)
                    if product is None or product.product_id in seen:
                        continue
                    if max_price is not None and product.price > max_price:
                        continue
                    seen.add(product.product_id)
                    products.append(product)
                    collected += 1
                    if collected >= quota:
                        break
                page += 1
                time.sleep(delay)
            category_collected += collected
        print(f"카테고리 {code}({category}): {category_collected}개 수집 (정렬 {'/'.join(SORT_CODES)} 합산)")
    return products


def _split_quota(total: int, parts: int) -> list[int]:
    """정렬 기준마다 최소 1개는 배정하면서 총합이 total을 넘지 않게 나눈다."""
    base = total // parts
    remainder = total % parts
    return [base + (1 if index < remainder else 0) for index in range(parts)]


def save_csv(products: list[CrawledProduct], csv_path: Path) -> None:
    fields = [
        "product_id", "name", "category", "color", "style", "purposes",
        "body_shapes", "price", "season", "stock", "url",
        "brand", "gender", "image_url", "image_path",
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for product in products:
            row = product.__dict__.copy()
            row["stock"] = "true" if product.stock else "false"
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="무신사 상품 수집기 (연구용 소규모)")
    parser.add_argument("--per-category", type=int, default=200, help="카테고리당 수집 개수(정렬 기준 합산, 정렬당 최대 100)")
    parser.add_argument("--delay", type=float, default=1.0, help="요청 간격(초)")
    parser.add_argument("--max-price", type=int, default=300_000, help="이 가격을 넘는 상품 제외")
    parser.add_argument("--skip-images", action="store_true", help="이미지 다운로드 생략")
    parser.add_argument(
        "--refresh-images", action="store_true",
        help="이미 캐시된 이미지도 다시 받는다(기존 저해상도 _500 캐시를 _big으로 갱신할 때 사용)",
    )
    parser.add_argument("--output", default="", help="CSV 저장 경로 (기본: data/products_musinsa.csv)")
    args = parser.parse_args()

    products = crawl(args.per_category, args.delay, args.max_price)
    if not args.skip_images:
        image_dir = GARMENT_RAW_DIR
        image_dir.mkdir(parents=True, exist_ok=True)
        for index, product in enumerate(products, start=1):
            try:
                download_image(product, image_dir, args.delay * 0.5, refresh=args.refresh_images)
            except Exception as exc:  # 이미지 한 장 실패로 전체를 멈추지 않는다.
                print(f"이미지 실패 {product.product_id}: {exc}")
            if index % 20 == 0:
                print(f"이미지 {index}/{len(products)}")

    csv_path = Path(args.output) if args.output else DATA_DIR / "products_musinsa.csv"
    save_csv(products, csv_path)
    print(f"총 {len(products)}개 상품을 {csv_path}에 저장했습니다.")


if __name__ == "__main__":
    main()
