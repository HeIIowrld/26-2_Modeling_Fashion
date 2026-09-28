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

    def test_measurement_text_is_escaped_and_zero_difference_is_displayed(self):
        source = (STATIC / "app.js").read_text(encoding="utf-8")
        function = source[source.index("function renderSizeFit("):source.index("function renderShoppingProducts(")]
        code = "const escapeHtml = (s) => String(s).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');\n" + function
        code += "\nconsole.log(JSON.stringify(renderSizeFit(" + json.dumps({
            "status": "compared", "summary": "M <script>bad()</script>", "closest_size": "M",
            "differences": [{"label": "가슴단면", "delta_cm": 0}],
            "columns": {"chest_width_cm": "가슴단면", "length_cm": "총장"},
            "size_options": [{"size": "M", "measurements": {"chest_width_cm": 54}, "available": None}],
        }) + ")));"
        html = self.run_js(code)
        self.assertNotIn("<script>", html)
        self.assertIn("&lt;script&gt;", html)
        self.assertIn("가슴단면 0cm", html)
        # 값이 없는 칸은 '—'로 채워 표의 칸이 밀리지 않게 한다.
        self.assertIn("<td>—</td>", html)
        # 2026-09-29: '판매 상태' 칸을 뺐다. 무신사 응답에 옵션별 재고가 없어
        # (available=None) 거의 모든 줄이 '재고 확인 필요'로 차 아무것도 알려 주지
        # 못했다. 재고 안내는 표 위 한 문장으로 남기고, 표에는 고른 사이즈를 짚어 준다.
        self.assertNotIn("재고 확인 필요", html)
        self.assertIn('<th scope="col">추천</th>', html)
        self.assertIn("<td>이 사이즈</td>", html)
        self.assertIn("해당 옵션의 재고는 상품 페이지에서 확인해주세요.", html)

    def test_pick_column_is_omitted_when_no_size_was_chosen(self):
        """추천 사이즈가 없으면 '추천' 칸은 빈 칸만 되므로 만들지 않는다."""
        source = (STATIC / "app.js").read_text(encoding="utf-8")
        function = source[source.index("function renderSizeFit("):source.index("function renderShoppingProducts(")]
        code = "const escapeHtml = (s) => String(s);\n" + function
        code += "\nconsole.log(JSON.stringify(renderSizeFit(" + json.dumps({
            "status": "compared", "summary": "실측만 있어요", "closest_size": "",
            "columns": {"chest_width_cm": "가슴단면"},
            "size_options": [{"size": "M", "measurements": {"chest_width_cm": 54}, "available": None}],
        }) + ")));"
        html = self.run_js(code)
        self.assertNotIn("추천", html)
        self.assertIn('<th scope="row">M</th>', html)

    def test_form_serializes_reference_dimensions_without_using_height_as_garment_length(self):
        source = (STATIC / "app.js").read_text(encoding="utf-8")
        function = source[source.index("function collectProfile("):source.index("/* ── 3단계:")]
        code = """
const $ = () => ({});
const state = {preferredColors: [], avoidedColors: [], preferredMaterials: []};
const wardrobeRowsWithImages = () => [];
class FormData {
  get(key) {return ({height_cm: '175', reference_top_chest_cm: '54.5', reference_top_length_cm: '70'})[key] ?? null;}
  getAll(key) {return key === 'change_categories' ? ['top', 'bottom', 'shoes'] : [];}
}
""" + function + "\nconsole.log(JSON.stringify(collectProfile()));"
        payload = self.run_js(code)
        self.assertEqual(payload["height_cm"], 175)
        self.assertEqual(payload["change_categories"], ["top", "bottom", "shoes"])
        self.assertEqual(payload["reference_measurements"]["top"], {"chest_width_cm": 54.5, "length_cm": 70})
        self.assertIsNone(payload["reference_measurements"]["bottom"]["length_cm"])


if __name__ == "__main__":
    unittest.main()
