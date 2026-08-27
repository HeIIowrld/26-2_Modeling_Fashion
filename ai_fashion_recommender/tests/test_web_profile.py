import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))  # 런타임 모듈은 src/에 있다
sys.path.insert(0, str(ROOT.parent / "web"))

from pipeline import build_profile, form_options


class BuildProfileTests(unittest.TestCase):
    def test_empty_payload_uses_dataclass_defaults(self):
        profile = build_profile({})
        self.assertEqual(profile.purpose, "데일리")
        self.assertEqual(profile.change_scope, "전체 변경")
        self.assertEqual(profile.budget, 150_000)
        self.assertEqual(profile.owned_items, [])

    def test_blank_weather_values_stay_none(self):
        profile = build_profile({"temperature_c": "", "humidity": None, "uv_index": 6})
        self.assertIsNone(profile.temperature_c)
        self.assertIsNone(profile.humidity)
        self.assertEqual(profile.uv_index, 6.0)

    def test_owned_items_without_color_are_dropped(self):
        profile = build_profile(
            {
                "owned_items": [
                    {"category": "bottom", "color": "네이비", "style": "캐주얼"},
                    {"category": "top", "color": "  "},
                ]
            }
        )
        self.assertEqual(len(profile.owned_items), 1)
        self.assertEqual(profile.owned_items[0].color, "네이비")
        self.assertEqual(profile.owned_items[0].category, "bottom")

    def test_owned_item_photo_placeholder_survives_until_image_analysis(self):
        profile = build_profile(
            {"owned_items": [{"category": "top", "image_index": 0}]}
        )
        self.assertEqual(len(profile.owned_items), 1)
        self.assertEqual(profile.owned_items[0].category, "top")
        self.assertEqual(profile.owned_items[0].color, "")

    def test_color_lists_drop_blank_entries(self):
        profile = build_profile({"preferred_colors": ["네이비", "  ", ""], "avoided_colors": []})
        self.assertEqual(profile.preferred_colors, ["네이비"])
        self.assertEqual(profile.avoided_colors, [])

    def test_material_categories_expand_to_existing_model_labels(self):
        profile = build_profile({"preferred_materials": ["면·일상 소재", "얇은 소재"]})
        self.assertEqual(
            profile.preferred_materials,
            ["코튼", "폴리에스터", "린넨", "쉬폰"],
        )

    def test_automatic_season_uses_a_concrete_season(self):
        self.assertIn(build_profile({"season": "자동"}).season, {"봄", "여름", "가을", "겨울"})


class FormOptionTests(unittest.TestCase):
    def test_required_style_options_use_plain_labels_and_engine_values(self):
        options = form_options()["styles"]
        labels = {option if isinstance(option, str) else option["label"] for option in options}
        self.assertEqual(labels, {"캐주얼", "미니멀", "스트릿", "클래식", "스포티", "기타"})

    def test_options_hide_the_no_change_scope(self):
        from recommendation_engine import CHANGE_SCOPE_MAP

        sent = set(form_options()["change_scopes"])
        self.assertEqual(sent, set(CHANGE_SCOPE_MAP) - {"현재 유지"})

    def test_silhouette_options_send_the_values_the_engine_expects(self):
        """화면이 보내는 값과 엔진이 아는 값이 어긋나면 체형 규칙이 조용히 잠든다."""
        from schemas import BODY_SHAPE_GOALS, GOAL_NONE, PROPORTION_GOALS, SILHOUETTE_GOALS

        sent = {option["value"] for option in form_options()["silhouette_goals"]}
        self.assertEqual(sent, set(SILHOUETTE_GOALS))
        # 목표를 고르지 않는 선택지 하나를 뺀 나머지는 전부 어떤 규칙이든 켜야 한다.
        self.assertEqual(sent - {GOAL_NONE}, BODY_SHAPE_GOALS | PROPORTION_GOALS)

    def test_every_silhouette_option_survives_a_round_trip(self):
        for option in form_options()["silhouette_goals"]:
            with self.subTest(goal=option["value"]):
                profile = build_profile({"silhouette_goal": option["value"]})
                self.assertEqual(profile.silhouette_goal, option["value"])

    def test_options_carry_a_label_for_every_value(self):
        for option in form_options()["silhouette_goals"]:
            with self.subTest(goal=option["value"]):
                self.assertTrue(option["label"])

    def test_stage_keys_match_pipeline_stages(self):
        from pipeline import STAGES

        self.assertEqual(
            [stage["key"] for stage in form_options()["stages"]],
            [key for key, _ in STAGES],
        )


if __name__ == "__main__":
    unittest.main()
