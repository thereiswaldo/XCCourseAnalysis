import unittest
import XCCourseHelper as XC
import pandas as pd
import numpy as np
from pandas.util.testing import assert_series_equal

class TestXCCourse(unittest.TestCase):

    def setUp(self):
        """ Setup for Pandas Tests """
        TestInputDirectory = './Course DataFrames/'
        TestFileName =  'Bayfront Park 7km.csv'
        try:
            data = pd.read_csv(TestInputDirectory + TestFileName,
                sep = ',',
                header = 0)
        except IOError:
            print('cannot open file')
        self.course = data

    def test_ChangeEvenToOdd(self):
        self.assertEqual(XC.ChangeEvenToOdd(1), 1)
        self.assertEqual(XC.ChangeEvenToOdd(2), 3)

    def test_KilometerToMeter(self):
        self.assertEqual(XC.KilometerToMeter(1), 1000)
        self.assertEqual(XC.KilometerToMeter(1.5), 1500)

    def test_SavitzkyGolayFilter(self):
        df = self.course
        DesiredOutput = pd.Series(index=[0, 1, 2, 3, 4], data=[81.596503, 81.622238, 81.644476, 81.663217, 81.678462], name='Elevation SavGol')
        ActualOutput = XC.SavitzkyGolayFilter(df, 10).iloc[0:5]
        assert_series_equal(DesiredOutput, ActualOutput)

    def test_positive_only(self):
        self.assertEqual(XC.positive_only(1), 1)
        self.assertEqual(XC.positive_only(-1), 0)

    def test_AboveOrEqualToThreshold(self):
        self.assertEqual(XC.AboveOrEqualToThreshold(1, 0.05), 1)
        self.assertEqual(XC.AboveOrEqualToThreshold(0.05, 0.05), 0.05)
        self.assertEqual(XC.AboveOrEqualToThreshold(-0.05, 0.05), 0)
        self.assertEqual(XC.AboveOrEqualToThreshold(0.05, 0.01), 0.05)
        self.assertEqual(XC.AboveOrEqualToThreshold(0.01, 0.01), 0.01)

    def test_myround(self):
        self.assertEqual(XC.myround(1.1551, prec=2, base=0.05), 1.15)
        self.assertEqual(XC.myround(1.155, prec=2, base=0.05), 1.15)
        self.assertEqual(XC.myround(1.14, prec=2, base=0.05), 1.15)
        self.assertEqual(XC.myround(1.1, prec=2, base=0.05), 1.10)

    def test_ElevationChange(self):
        df = self.course
        DesiredOutput = pd.Series(index=[0, 1, 2, 3, 4], data=[np.nan, 0.02, 0.02, 0.02, 0.02], name='Elevation')
        ActualOutput = XC.ElevationChange(df['Elevation']).iloc[0:5]
        assert_series_equal(DesiredOutput, ActualOutput)
        
    def test_ElevationGain(self):
        df = self.course
        DesiredOutput = 44.84333333333
        ActualOutput = XC.ElevationGain(XC.ElevationChange(df['Elevation']))
        self.assertAlmostEqual(DesiredOutput, ActualOutput)

if __name__ == '__main__':
    unittest.main()