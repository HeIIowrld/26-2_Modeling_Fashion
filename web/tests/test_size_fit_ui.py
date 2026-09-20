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
        result = subprocess.run(["node", "-e", code], text=True, capture_output=True, check=True)
        return json.loads(result.stdout)

    def test_measurement_text_is_escaped_and_zero_difference_is_displayed(self):
        source = (STATIC / "app.js").read_text()
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
        self.assertIn("재고 확인 필요", html)
        self.assertIn("<td>—</td>", html)

    def test_form_serializes_reference_dimensions_without_using_height_as_garment_length(self):
        source = (STATIC / "app.js").read_text()
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
