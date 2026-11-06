import tempfile
import unittest
from pathlib import Path

import numpy as np

from points_utils import MAX_POINTS, read_points_csv, validate_xy


class PointInputTests(unittest.TestCase):
    def test_reads_bundled_sample(self):
        sample = Path(__file__).parents[1] / "xy_locs.csv"
        xy = read_points_csv(sample)
        self.assertEqual(xy.shape, (10, 2))
        self.assertTrue(np.isfinite(xy).all())

    def test_rejects_malformed_headers_and_values(self):
        for content in ("x\n1\n", "y,x\n1,2\n", "x,y\n1,nope\n", "x,y\n1,\n"):
            with self.subTest(content=content), tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "points.csv"
                path.write_text(content, encoding="utf-8")
                with self.assertRaises(ValueError):
                    read_points_csv(path)

    def test_validates_shape_and_limits(self):
        with self.assertRaisesRegex(ValueError, "exactly two"):
            validate_xy([1, 2])
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_xy([[1, np.inf]])
        with self.assertRaisesRegex(ValueError, "no more than"):
            validate_xy(np.zeros((MAX_POINTS + 1, 2)))


if __name__ == "__main__":
    unittest.main()
