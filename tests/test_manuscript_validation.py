from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "paper" / "validate_manuscript.py"


class ManuscriptValidationTests(unittest.TestCase):
    def test_manuscript_validator_passes(self) -> None:
        result = subprocess.run(
            [sys.executable, str(VALIDATOR)],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("manuscript validation passed", result.stdout)


if __name__ == "__main__":
    unittest.main()
