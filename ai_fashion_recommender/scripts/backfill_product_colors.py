"""이미 만들어 둔 카탈로그의 색 칼럼만 현재 규격으로 다시 채운다.

전체 보강(enrich_catalog.py)은 상품 사진과 GPU 모델이 필요하다. 색은 옵션 API 하나로
끝나므로, 이미 있는 CSV 의 색 칼럼만 갱신한다. 다른 칼럼은 건드리지 않는다.

왜 필요한가(2026-09-25 표본 60개 실측):
- 색 옵션이 있는 상품의 68%가 2색 이상인데 대표 색 하나만 저장돼 있었다.
- 11%는 저장된 색이 실제 판매 색에 아예 없었다('황토색' 상품이 '블루'로 저장).
색은 체형 규칙(preferred_top_colors)과 회피 색 조건이 직접 보는 값이다.

색 출처의 우선순위는 대표 사진과 얼마나 맞는지로 정했다(2026-09-25):
  1. 상품명 색  — 상의 81개에서 78%, 내가 확신한 65개에서 85%
  2. 모든 컬러칩이 하나의 팔레트인 경우만 대표 색으로 사용한다.
     첫 컬러칩만 고르면 대표 사진과 일치율이 46%여서 다색 옵션은 보류한다.
  3. 둘 다 없으면 **비워 둔다**. 예전 크롤러는 '그레이'를 넣었고 근거 없는 그레이가 329개였다.

상품명 색은 네트워크가 필요 없다. 옵션 API 갱신은 --fetch 를 줄 때만 한다.

usage:
    python scripts/backfill_product_colors.py                     # 제목·기존 칩으로 재계산
    python scripts/backfill_product_colors.py --fetch             # 색 옵션도 새로 받는다
    python scripts/backfill_product_colors.py --limit 20 --dry-run
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import DATA_DIR  # noqa: E402
from enrich_catalog import load_derivation  # noqa: E402
from product_colors import palette_of, palettes_for, title_palettes  # noqa: E402
from product_measurements import color_options_from  # noqa: E402
from shopping_http import fetch_json  # noqa: E402

OPTIONS_URL = "https://goods-detail.musinsa.com/api2/goods/{no}/options"
NEW_COLUMNS = ("color_options", "detail_colors", "color_evidence")


def representative_color(title, names, table):
    """여러 색 중 첫 색을 대표 사진 색으로 단정하지 않는다."""
    colors = title_palettes(title, table)
    if len(colors) == 1:
        return colors[0], "product_title"
    # 제목에 두 색이 있으면 배색/팩 상품일 수 있어 단일 색 옵션보다 우선 보류한다.
    if colors:
        return "", "multiple_title_colors"
    palettes = palettes_for(names, table)
    if (len(palettes) == 1 and names
            and all(palette_of(name, table) == palettes[0] and len(title_palettes(name, table)) <= 1
                    for name in names)):
        return palettes[0], "single_option_palette"
    return "", "ambiguous_options" if names else "no_color_evidence"


class RateLimited(Exception):
    """무신사가 429를 돌려줬다. 실패를 조용히 삼키면 '색 없는 상품'과 구분되지 않는다."""


def fetch_colors(product_id: str, timeout: float) -> list[str]:
    number = product_id.removeprefix("MS")
    if not number.isdigit():
        return []
    try:
        return color_options_from(fetch_json(OPTIONS_URL.format(no=number), timeout))
    except urllib.error.HTTPError as exc:
        if exc.code == 429:
            raise RateLimited(product_id) from exc
        raise


def main() -> int:
    cli = argparse.ArgumentParser()
    cli.add_argument("--input", type=Path, default=DATA_DIR / "products_musinsa_enriched.csv")
    cli.add_argument("--output", type=Path)
    cli.add_argument("--limit", type=int)
    cli.add_argument("--fetch", action="store_true", help="색 옵션을 무신사에서 새로 받는다")
    cli.add_argument("--missing-only", action="store_true", help="색이 비어 있는 행만 갱신한다")
    cli.add_argument("--preserve-existing-colors", action="store_true", help="전체 색 옵션을 보강하되 기존 대표 색은 유지한다")
    cli.add_argument("--from-measurement-cache", type=Path, help="실측 수집 시 받은 색 옵션을 네트워크 없이 재사용한다")
    cli.add_argument("--delay", type=float, default=0.25, help="요청 간격(초)")
    cli.add_argument("--retries", type=int, default=3)
    cli.add_argument("--backoff", type=float, default=30.0, help="429 뒤 첫 대기(초)")
    cli.add_argument("--timeout", type=float, default=5.0)
    cli.add_argument("--max-failure-rate", type=float, default=0.10)
    cli.add_argument("--dry-run", action="store_true")
    cli.add_argument("--force", action="store_true", help="실패가 많아도 저장")
    args = cli.parse_args()
    if args.fetch and args.from_measurement_cache:
        cli.error("--fetch와 --from-measurement-cache는 함께 사용할 수 없습니다.")

    with args.input.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    for column in NEW_COLUMNS:
        if column not in fields:
            fields.append(column)
    targets = [row for row in rows if not args.missing_only or not row.get("color", "").strip()]
    targets = targets[:args.limit] if args.limit else targets
    table = load_derivation()["musinsa_color"]

    started = time.monotonic()
    fetched, failures, rate_limited = [], 0, 0
    if args.from_measurement_cache:
        for row in targets:
            try:
                saved = json.loads((args.from_measurement_cache / f"{row['product_id']}.json").read_text(encoding="utf-8"))
                record = saved["record"]
                if record.get("product_id") != row["product_id"] or "options_unavailable" in record.get("issues", []):
                    raise ValueError("색 옵션 조회 미완료")
                raw_options = (saved.get("raw") or {}).get("options")
                fetched.append(color_options_from({"data": raw_options}) if isinstance(raw_options, dict)
                               else record.get("color_options", []))
            except (OSError, ValueError, KeyError, TypeError):
                fetched.append(None)
                failures += 1
    elif not args.fetch:
        # 상품명 색만으로도 대부분 채워진다. 기존 CSV 의 칩 값을 그대로 쓴다.
        fetched = [[value.strip() for value in (row.get("detail_colors") or row.get("detail_color") or "").split("|")
                    if value.strip()] for row in targets]
    # 무신사는 몰아치면 429를 준다(2026-09-25 실측: 4스레드·무간격 2224건에서 94% 거절).
    # 한 줄씩 간격을 두고 받고, 429 가 나오면 기다렸다 다시 시도한다.
    for index, row in enumerate(targets) if args.fetch else ():
        names, wait = None, args.backoff
        for attempt in range(args.retries + 1):
            try:
                names = fetch_colors(row["product_id"], args.timeout)
                break
            except RateLimited:
                rate_limited += 1
                if attempt == args.retries:
                    failures += 1
                    break
                print(f"  429: {wait:.0f}초 쉬고 다시 시도합니다 ({row['product_id']})", flush=True)
                time.sleep(wait)
                wait *= 2
            except (OSError, ValueError, TypeError, KeyError):
                failures += 1
                break
        fetched.append(names)
        if index % 200 == 199:
            print(f"  {index + 1}/{len(targets)} 진행 (실패 {failures}, 429 {rate_limited})", flush=True)
        time.sleep(args.delay)

    changed = multi = emptied = from_title = 0
    for row, names in zip(targets, fetched):
        if names is None:
            continue  # 조회 실패는 '색 없음'이 아니다. 기존 근거를 보존한다.
        palettes = palettes_for(names, table)
        before = row.get("color", "")
        if names:
            row["detail_color"] = names[0]
            row["detail_colors"] = "|".join(names)
        color, evidence = representative_color(row.get("name", ""), names, table)
        if args.preserve_existing_colors and before.strip():
            color, evidence = before, row.get("color_evidence") or "existing_catalog"
        row["color"] = color
        row["color_evidence"] = evidence
        row["color_options"] = "|".join(palettes) if palettes else color
        from_title += evidence == "product_title"
        changed += row["color"] != before
        multi += len(palettes) > 1
        emptied += not row["color"]
    elapsed = time.monotonic() - started

    print(f"대상 {len(targets)}개 / 색 옵션 확보 {sum(bool(n) for n in fetched)}개 "
          f"({elapsed:.1f}초, 상품당 {elapsed / max(1, len(targets)):.3f}초)")
    print(f"  대표 색이 바뀐 상품 {changed}  상품명에서 정한 색 {from_title}  "
          f"여러 색을 채운 상품 {multi}  근거가 없어 비운 상품 {emptied}")
    failure_rate = failures / max(1, len(targets))
    print(f"  못 받은 상품 {failures} ({failure_rate:.1%}), 429 응답 {rate_limited}회")
    if failure_rate > args.max_failure_rate and not (args.dry_run or args.force):
        # 거절당한 상품과 '색 옵션이 없는 상품'은 결과가 똑같이 비어 보인다.
        # 이 상태로 저장하면 치우친 일부만 갱신된 카탈로그가 남는다.
        print("실패가 많아 저장하지 않습니다. 잠시 뒤 다시 실행하세요(--force 로 무시 가능).")
        return 1
    if args.dry_run:
        for row in targets[:10]:
            print(f"  {row['product_id']} color={row.get('color', '')!r} "
                  f"options={row.get('color_options', '')!r} ← {row.get('detail_colors', '')!r}")
        return 0

    output = args.output or args.input
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    print(f"저장: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
