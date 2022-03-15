import unittest
import GradeAdjustedPace
import pandas as pd
import numpy as np
from pandas.util.testing import assert_series_equal, assert_frame_equal


class TestGradeAdjustedPace(unittest.TestCase):
    def setUp(self):
        self.HillDF = pd.DataFrame(
            {
                "Hill Height": [1, 2, 3],
                "Hill Distance": [1, 2, 3],
                "Hill Grade": [0.5, 2, 3],
            }
        )
        self.GAP = GradeAdjustedPace.GAP(self.HillDF)

    def test_gradientCalculation(self):
        self.assertEqual(self.HillDF, self.GAP.gradientCalculation())
