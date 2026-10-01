"""상품 실측 UI의 실제 렌더링 함수와 입력 직렬화를 Node에서 검사한다."""

import json
import shutil
import subprocess
import unittest
from pathlib import Path


STATIC = Path(__file__).resolve().parents[1] / "static"


@unittest.skipUnless(shutil.which("node"), "Node is required for JavaScript behavior tests")
class SizeFitUITests(unittest.TestCase):
    def run_js(self, code):
        # Node 는 UTF-8 로 출력한다. text=True 만 쓰면 Windows 에서 cp949 로 읽어 한글이 깨진다.
        result = subprocess.run(["node", "-e", code], text=True, encoding="utf-8",
                                capture_output=True, check=True)
        return json.loads(result.stdout)

    def test_catalog_card_distinguishes_unknown_stock_date(self):
        source = (STATIC / "app.js").read_text(encoding="utf-8")
        function = source[source.index("function renderShoppingProductCard("):source.index("function renderShoppingProducts(")]
        setup = """
const escapeHtml = (s) => String(s).replaceAll('<', '&lt;');
const state = {shoppingSelection: {}, tryon: {available: false}};
const renderSizeFit = () => '';
const renderShoppingReason = () => '';
const product = {product_id: 'MS1', name: '셔츠', category: 'top', price: 100,
                 url: '', image_url: '', source: 'musinsa_catalog_fallback'};
"""
        code = setup + function + """
console.log(JSON.stringify([
  renderShoppingProductCard(product),
  renderShoppingProductCard({...product, stock_checked_at: '2026-09-25T12:00:00Z'}),
  renderShoppingProductCard({...product, source: 'musinsa_live'})
]));
"""
        unknown, dated, live = self.run_js(code)
        self.assertIn("재고 확인 시점 미상", unknown)
        self.assertIn("수집 기준", dated)
        self.assertNotIn("시점 미상", dated)
        self.assertNotIn("시점 미상", live)

    def _render_size_fit(self, fit, escape=True):
        source = (STATIC / "app.js").read_text(encoding="utf-8")
        function = source[source.index("function renderSizeFit("):
                          source.index("function renderShoppingProductCard(")]
        helper = ("const escapeHtml = (s) => String(s).replaceAll('&', '&amp;')"
                  ".replaceAll('<', '&lt;').replaceAll('>', '&gt;');"
                  if escape else "const escapeHtml = (s) => String(s);")
        code = (helper + "\n" + function
                + "\nconsole.log(JSON.stringify(renderSizeFit(" + json.dumps(fit) + ")));")
        return self.run_js(code)

    def test_size_table_renders_rows_and_escapes_text(self):
        """기준 옷 비교를 없앤 뒤(2026-10-01) 남은 것은 상품 실측표뿐이다.

        비교·추천 사이즈·기준 옷 행이 모두 사라졌는지, 값이 이스케이프되는지 본다.
        """
        html = self._render_size_fit({
            "status": "available",
            "columns": {"chest_width_cm": "가슴단면", "length_cm": "총장"},
            "size_options": [
                {"size": "M <script>bad()</script>", "measurements": {"chest_width_cm": 54}},
                {"size": "L", "measurements": {"chest_width_cm": 58, "length_cm": 72}},
            ],
        })
        self.assertNotIn("<script>", html)
        self.assertIn("&lt;script&gt;", html)
        self.assertIn("<td>54</td>", html)
        # 값이 없는 칸은 '—'로 채워 표의 칸이 밀리지 않게 한다.
        self.assertIn("<td>—</td>", html)
        # 기준 옷 비교의 흔적이 남아 있으면 안 된다.
        for gone in ("기준 옷", "이 사이즈", "사이즈 비교"):
            self.assertNotIn(gone, html)

    def test_no_table_renders_nothing(self):
        """실측이 없다고 해서 빈 카드를 그리지 않는다."""
        self.assertEqual(self._render_size_fit(
            {"status": "missing_measurements", "columns": {}, "size_options": []},
            escape=False), "")
