"""브라우저 버튼만으로 공용 핏 데이터 CSV를 만드는 CPU 라벨링 도구.

예:
  python scripts/label_fit_vision.py \
    --images-dir data/fit_vision/images/shop_bottom \
    --dataset-root data/fit_vision \
    --output-csv data/fit_vision/annotations.csv \
    --category bottom --source-domain shop
"""

from __future__ import annotations

import argparse
import html
import mimetypes
import sys
import threading
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fit_labeling import FitAnnotationStore, discover_images, infer_group_id
from fit_vision_schema import FIT_VISION_TASKS


TASK_LABELS = {
    "upper_fit": "상의 핏",
    "upper_length": "상의 기장",
    "bottom_silhouette": "하의 핏",
    "bottom_length": "하의 기장",
    "quality": "판정 품질",
}


def page_html(state, index: int, message: str = "") -> bytes:
    image = state["images"][index]
    category = state["category"]
    existing = state["store"].get(image, category) or {}
    tasks = (
        ("upper_fit", "upper_length", "quality")
        if category == "top" else ("bottom_silhouette", "bottom_length", "quality")
    )
    completed = sum(
        state["store"].get(path, category) is not None for path in state["images"]
    )
    sections = []
    for task in tasks:
        buttons = [
            '<label class="choice"><input type="radio" name="{0}" value="">모름</label>'.format(task)
        ]
        for number, label in enumerate(FIT_VISION_TASKS[task].labels, 1):
            checked = " checked" if existing.get(task, "") == label else ""
            buttons.append(
                f'<label class="choice"><input type="radio" name="{task}" '
                f'value="{html.escape(label)}"{checked}>{number}. {html.escape(label)}</label>'
            )
        sections.append(
            f'<section><h2>{TASK_LABELS[task]}</h2><div class="choices">{"".join(buttons)}</div></section>'
        )
    next_index = min(index + 1, len(state["images"]) - 1)
    previous_index = max(0, index - 1)
    document = f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>FITTA 핏 라벨링</title>
<style>
body{{margin:0;background:#f3f4f6;font-family:-apple-system,BlinkMacSystemFont,sans-serif;color:#171717}}
main{{max-width:1100px;margin:auto;padding:20px}} .top{{display:flex;justify-content:space-between;gap:12px;align-items:center}}
.panel{{display:grid;grid-template-columns:minmax(320px,1.2fr) minmax(320px,1fr);gap:18px}}
.photo,.form{{background:white;border-radius:20px;padding:16px;box-shadow:0 8px 30px #0001}}
.photo img{{width:100%;max-height:72vh;object-fit:contain;background:#eee;border-radius:12px}}
h1{{font-size:24px}} h2{{font-size:16px;margin:16px 0 8px}} .choices{{display:flex;flex-wrap:wrap;gap:8px}}
.choice{{border:1px solid #ddd;border-radius:999px;padding:9px 13px;cursor:pointer}} .choice:has(input:checked){{background:#171717;color:white}}
.choice input{{display:none}} button,.nav{{border:0;border-radius:12px;padding:12px 16px;text-decoration:none;display:inline-block}}
button{{width:100%;background:#2563eb;color:white;font-size:16px;font-weight:700;margin-top:18px;cursor:pointer}}
.nav{{background:white;color:#222}} .muted{{color:#666;font-size:13px}} .message{{color:#166534}}
@media(max-width:780px){{.panel{{grid-template-columns:1fr}}}}
</style></head><body><main>
<div class="top"><div><h1>FITTA 핏 라벨링</h1><div class="muted">{completed}/{len(state['images'])} 완료 · {html.escape(category)} · {html.escape(state['source_domain'])}</div></div>
<div><a class="nav" href="/?index={previous_index}">이전</a> <a class="nav" href="/?index={next_index}">다음</a></div></div>
<p class="message">{html.escape(message)}</p>
<div class="panel"><div class="photo"><img src="/image?index={index}"><p class="muted">{html.escape(image.name)}</p></div>
<form class="form" method="post" action="/save"><input type="hidden" name="index" value="{index}">
{"".join(sections)}
<button type="submit">저장하고 다음 사진</button></form></div>
</main></body></html>"""
    return document.encode("utf-8")


class LabelHandler(BaseHTTPRequestHandler):
    state = None

    def log_message(self, format, *args):  # noqa: A003
        return

    def _index(self, query) -> int:
        try:
            return max(0, min(int(query.get("index", [0])[0]), len(self.state["images"]) - 1))
        except (TypeError, ValueError):
            return 0

    def do_GET(self):  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(parsed.query)
        index = self._index(query)
        if parsed.path == "/image":
            path = self.state["images"][index]
            data = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "image/jpeg")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        body = page_html(self.state, index)
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):  # noqa: N802
        if self.path != "/save":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        form = urllib.parse.parse_qs(self.rfile.read(length).decode("utf-8"), keep_blank_values=True)
        index = self._index(form)
        image = self.state["images"][index]
        category = self.state["category"]
        tasks = (
            ("upper_fit", "upper_length", "quality")
            if category == "top" else ("bottom_silhouette", "bottom_length", "quality")
        )
        labels = {task: form.get(task, [""])[0] for task in tasks if form.get(task, [""])[0]}
        labels.setdefault("quality", "판정 가능")
        relative = image.relative_to(self.state["images_dir"])
        mask_path = None
        if self.state["mask_dir"]:
            candidate = (self.state["mask_dir"] / relative).with_suffix(".png")
            mask_path = candidate if candidate.is_file() else None
        try:
            group_id = infer_group_id(
                image, self.state["source_domain"], self.state["group_regex"]
            )
            self.state["store"].save(
                image,
                category=category,
                source_domain=self.state["source_domain"],
                group_id=group_id,
                labels=labels,
                mask_path=mask_path,
            )
        except ValueError as error:
            body = page_html(self.state, index, f"저장 실패: {error}")
            self.send_response(400)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(body)
            return
        next_index = min(index + 1, len(self.state["images"]) - 1)
        self.send_response(303)
        self.send_header("Location", f"/?index={next_index}")
        self.end_headers()


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU에서 실행하는 FITTA 핏 라벨링 페이지")
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--category", required=True, choices=("top", "bottom"))
    parser.add_argument("--source-domain", required=True, choices=("user", "shop"))
    parser.add_argument("--mask-dir")
    parser.add_argument("--group-regex", default="")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    images_dir = Path(args.images_dir).expanduser().resolve()
    images = discover_images(images_dir)
    if not images:
        raise SystemExit(f"라벨링할 JPG/PNG/WEBP가 없습니다: {images_dir}")
    state = {
        "images": images,
        "images_dir": images_dir,
        "dataset_root": Path(args.dataset_root).expanduser().resolve(),
        "store": FitAnnotationStore(args.output_csv, args.dataset_root),
        "category": args.category,
        "source_domain": args.source_domain,
        "mask_dir": Path(args.mask_dir).expanduser().resolve() if args.mask_dir else None,
        "group_regex": args.group_regex,
    }
    LabelHandler.state = state
    server = ThreadingHTTPServer(("127.0.0.1", args.port), LabelHandler)
    url = f"http://127.0.0.1:{args.port}"
    print(f"핏 라벨링 페이지: {url}")
    print(f"사진 {len(images)}장 · 저장: {Path(args.output_csv).resolve()}")
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
