import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_vton_shoulders import shoulder_gate, shoulder_width  # noqa: E402


class ShoulderAuditTests(unittest.TestCase):
    def test_measurement_and_visibility(self):
        self.assertAlmostEqual(shoulder_width({
            "left_shoulder": (0.7, 0.2, 0.9),
            "right_shoulder": (0.4, 0.2, 0.9),
        }), 0.3)
        self.assertIsNone(shoulder_width({
            "left_shoulder": (0.7, 0.2, 0.3),
            "right_shoulder": (0.4, 0.2, 0.9),
        }))

    def test_guard_rejects_additional_shrinkage(self):
        self.assertTrue(shoulder_gate(0.288, 0.3))
        self.assertFalse(shoulder_gate(0.287, 0.3))
        self.assertFalse(shoulder_gate(None, 0.3))
        with self.assertRaises(ValueError):
            shoulder_gate(0.3, 0.3, 0.5)


if __name__ == "__main__":
    unittest.main()
