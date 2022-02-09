import unittest
import XCCourseHelper as XC
import pandas as pd
import numpy as np
from pandas.util.testing import assert_series_equal

class TestXCCourse(unittest.TestCase):

    def setUp(self):
        Bayfront =  'Bayfront Park 7km'
        self.BayfrontCourse = XC.ReadCourseDataFrame(Bayfront)
        Western10km =  'Western10km'
        self.Western10km = XC.ReadCourseDataFrame(Western10km)
        FortHenry10km = 'FortHenry10km'
        self.FortHenry10km = XC.ReadCourseDataFrame(FortHenry10km)

    def test_ChangeEvenToOdd(self):
        self.assertEqual(XC.ChangeEvenToOdd(1), 1)
        self.assertEqual(XC.ChangeEvenToOdd(2), 3)

    def test_KilometerToMeter(self):
        self.assertEqual(XC.KilometerToMeter(1), 1000)
        self.assertEqual(XC.KilometerToMeter(1.5), 1500)

    def test_SavitzkyGolayFilter(self):
        df = self.BayfrontCourse
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
        df = self.BayfrontCourse
        DesiredOutput = pd.Series(index=[0, 1, 2, 3, 4], data=[np.nan, 0.02, 0.02, 0.02, 0.02], name='Elevation')
        ActualOutput = XC.ElevationChange(df['Elevation']).iloc[0:5]
        assert_series_equal(DesiredOutput, ActualOutput)
        
    def test_ElevationGainFromChange(self):
        # Bayfront
        DesiredOutput = 66.4 #strava 49m
        ActualOutput = round(XC.ElevationGainFromChange(XC.ElevationChange(self.BayfrontCourse['Elevation'])),1)
        self.assertAlmostEqual(DesiredOutput, ActualOutput)
        # Western10km
        DesiredOutput = 113.2 # Strava 112-117m
        ActualOutput = round(XC.ElevationGainFromChange(XC.ElevationChange(self.Western10km['Elevation'])),1)
        self.assertAlmostEqual(DesiredOutput, ActualOutput)
        #Fort Henry 10km
        DesiredOutput = 21.8 # Strava 33m
        ActualOutput = round(XC.ElevationGainFromChange(XC.ElevationChange(self.FortHenry10km['Elevation'])),1)
        self.assertAlmostEqual(DesiredOutput, ActualOutput)

    def test_GetElevationGain(self):
        #Fort Henry 10km
        DesiredOutput = 21.8 # Strava 33m
        ActualOutput = round(XC.GetElevationGain(self.FortHenry10km['Elevation']),1)
        self.assertAlmostEqual(DesiredOutput, ActualOutput)

    def test_ReadCourseDataFrame(self):
        BayfrontDF = self.BayfrontCourse
        self.assertEqual(BayfrontDF.shape[0], 9900)
        self.assertEqual(BayfrontDF.shape[1], 16)
        Western10kmDF = self.Western10km
        self.assertEqual(Western10kmDF.shape[0], 10279)
        self.assertEqual(Western10kmDF.shape[1], 16)
        FortHenry10km = self.FortHenry10km
        self.assertEqual(FortHenry10km.shape[0], 10495)
        self.assertEqual(FortHenry10km.shape[1], 16)

if __name__ == '__main__':
    unittest.main()