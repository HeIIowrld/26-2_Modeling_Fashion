import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from schemas import OutfitAnalysis


class OutfitSummaryTests(unittest.TestCase):
    def test_summary_prioritizes_clear_upper_and_lower_conclusion(self):
        result = OutfitAnalysis(
            parser_backend="test",
            upper_color="화이트",
            lower_color="네이비",
            color_harmony="안정적인 무채색 조합",
            detected_items=["top", "pants"],
            style="캐주얼",
            upper_type="폴로 셔츠",
            lower_type="팬츠",
            lower_subtype="분석 보류",
            pant_leg_shape="분석 보류",
            pant_length="풀렝스",
            sleeve_length="반팔",
            lower_fit="와이드핏 추정",
            lower_details=["포켓"],
            attribute_sources={"lower_fit": "mask"},
        )

        self.assertEqual(result.to_summary_dict(), {
            "상의": "화이트 폴로 셔츠 (반팔)",
            "하의": "네이비 팬츠 (풀렝스, 포켓)",
        })

    def test_summary_uses_accepted_detailed_bottom_attributes(self):
        result = OutfitAnalysis(
            parser_backend="test",
            upper_color="화이트",
            lower_color="블루",
            color_harmony="안정적인 무채색 조합",
            detected_items=["top", "pants"],
            style="캐주얼",
            upper_type="셔츠",
            lower_type="청바지",
            lower_subtype="청바지",
            pant_leg_shape="스트레이트",
            pant_length="풀렝스",
            sleeve_length="긴팔",
            lower_details=["5포켓"],
        )

        summary = result.to_summary_dict()
        self.assertEqual(summary["상의"], "화이트 셔츠 (긴팔)")
        self.assertEqual(summary["하의"], "블루 청바지 (스트레이트, 풀렝스, 5포켓)")

    def test_summary_exposes_layering_and_rolled_sleeves(self):
        result = OutfitAnalysis(
            parser_backend="test",
            upper_color="네이비",
            lower_color="베이지",
            color_harmony="안정적인 무채색 조합",
            detected_items=["top", "pants"],
            style="캐주얼",
            upper_type="니트",
            lower_type="팬츠",
            sleeve_length="긴팔",
            visible_sleeve_length="7부 소매",
            sleeve_state="걷음 가능성 높음",
            layering_state="레이어드",
            upper_items=["셔츠", "니트"],
            inner_category="셔츠",
            outer_category="니트",
        )

        self.assertEqual(
            result.to_summary_dict()["상의"],
            "네이비 셔츠 + 니트 (긴팔·소매 걷음)",
        )


if __name__ == "__main__":
    unittest.main()
