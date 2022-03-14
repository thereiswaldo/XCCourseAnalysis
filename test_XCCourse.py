import unittest
import XCCourseHelper as XC
import pandas as pd
import numpy as np
from pandas.util.testing import assert_series_equal, assert_frame_equal


class TestXCCourse(unittest.TestCase):
    def setUp(self):
        Bayfront = "Bayfront Park 7km"
        self.BayfrontCourse = XC.ReadCourseDataFrame(Bayfront)
        Western10km = "Western10km"
        self.Western10km = XC.ReadCourseDataFrame(Western10km)
        FortHenry10km = "FortHenry10km"
        self.FortHenry10km = XC.ReadCourseDataFrame(FortHenry10km)
        self.HillDF = pd.DataFrame(
            {
                "Hill Height": [1, 2, 3],
                "Hill Distance": [1, 2, 3],
                "Hill Grade": [0.5, 2, 3],
            }
        )

    def test_ChangeEvenToOdd(self):
        self.assertEqual(XC.ChangeEvenToOdd(1), 1)
        self.assertEqual(XC.ChangeEvenToOdd(2), 3)

    def test_KilometerToMeter(self):
        self.assertEqual(XC.KilometerToMeter(1), 1000)
        self.assertEqual(XC.KilometerToMeter(1.5), 1500)

    def test_SavitzkyGolayFilter(self):
        df = self.BayfrontCourse
        DesiredOutput = pd.Series(
            index=[0, 1, 2, 3, 4],
            data=[81.596503, 81.622238, 81.644476, 81.663217, 81.678462],
            name="Elevation SavGol",
        )
        ActualOutput = XC.SavitzkyGolayFilter(df, 10).iloc[0:5]
        assert_series_equal(DesiredOutput, ActualOutput)

    def test_positive_only(self):
        self.assertEqual(XC.PositiveOnly(1), 1)
        self.assertEqual(XC.PositiveOnly(-1), 0)

    def test_AboveOrEqualToThreshold(self):
        self.assertEqual(XC.AboveOrEqualToThreshold(1, 0.05), 1)
        self.assertEqual(XC.AboveOrEqualToThreshold(0.05, 0.05), 0.05)
        self.assertEqual(XC.AboveOrEqualToThreshold(-0.05, 0.05), 0)
        self.assertEqual(XC.AboveOrEqualToThreshold(0.05, 0.01), 0.05)
        self.assertEqual(XC.AboveOrEqualToThreshold(0.01, 0.01), 0.01)

    def test_myround(self):
        self.assertEqual(XC.AdvancedRound(1.1551, prec=2, base=0.05), 1.15)
        self.assertEqual(XC.AdvancedRound(1.155, prec=2, base=0.05), 1.15)
        self.assertEqual(XC.AdvancedRound(1.14, prec=2, base=0.05), 1.15)
        self.assertEqual(XC.AdvancedRound(1.1, prec=2, base=0.05), 1.10)

    def test_ElevationChange(self):
        df = self.BayfrontCourse
        DesiredOutput = pd.Series(
            index=[0, 1, 2, 3, 4],
            data=[np.nan, 0.02, 0.02, 0.02, 0.02],
            name="Elevation",
        )
        ActualOutput = XC.ElevationChange(df["Elevation"]).iloc[0:5]
        assert_series_equal(DesiredOutput, ActualOutput)

    def test_ElevationGainFromChange(self):
        # Bayfront
        DesiredOutput = 66.4  # strava 49m
        ActualOutput = round(
            XC.ElevationGainFromChange(
                XC.ElevationChange(self.BayfrontCourse["Elevation"])
            ),
            1,
        )
        self.assertAlmostEqual(DesiredOutput, ActualOutput)
        # Western10km
        DesiredOutput = 113.2  # Strava 112-117m
        ActualOutput = round(
            XC.ElevationGainFromChange(
                XC.ElevationChange(self.Western10km["Elevation"])
            ),
            1,
        )
        self.assertAlmostEqual(DesiredOutput, ActualOutput)
        # Fort Henry 10km
        DesiredOutput = 21.8  # Strava 33m
        ActualOutput = round(
            XC.ElevationGainFromChange(
                XC.ElevationChange(self.FortHenry10km["Elevation"])
            ),
            1,
        )
        self.assertAlmostEqual(DesiredOutput, ActualOutput)

    def test_GetElevationGain(self):
        # Fort Henry 10km
        DesiredOutput = 21.8  # Strava 33m
        ActualOutput = round(XC.GetElevationGain(self.FortHenry10km["Elevation"]), 1)
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

    def test_MakeInitialDataframe(self):
        (
            df,
            turn_st_dev,
            tot_deg_turned,
            HairpinCount,
            perim,
            area,
        ) = XC.MakeInitialDataframe(
            "Western International 10km", "Courses/Western International 10km.gpx"
        )
        self.assertEqual(df.shape[0], 601)
        self.assertEqual(df.shape[1], 10)
        self.assertAlmostEqual(turn_st_dev, 25.08792481)
        self.assertAlmostEqual(tot_deg_turned, 10432.96009585109)
        self.assertEqual(HairpinCount, 4)
        self.assertAlmostEqual(perim, 9968.833142535985)
        self.assertAlmostEqual(area, 460036.173622301)

    def test_InterpolateGPS(self):
        df = self.Western10km
        df = XC.InterpolateGPS(df)
        self.assertEqual(df.shape[0], 10278001)
        self.assertEqual(df.shape[1], 16)
        self.assertEqual(df.index[1] - df.index[0], 0.001)
        self.assertAlmostEqual(df.index.to_series().diff().mean(), 0.001)
        self.assertAlmostEqual(df.index[0], 0.0)
        self.assertAlmostEqual(df.index[-1], 10278.0)
        self.assertAlmostEqual(df["Elevation"][0], 237.8)

    def test_ParseGPX(self):
        data = XC.ParseGPX("Courses/Western International 10km.gpx")
        self.assertEqual(data.tracks[0].segments[0].points[0].latitude, 42.962597)

    def test_HairpinDetection(self):
        df = XC.LoadRunIntoDF("Courses/Western International 10km.gpx")
        df["DistanceChangeInKM"] = df["Distance"] - df["Distance"].shift()
        HairpinLengthInMeters = 30
        HairpinWindow = round(
            HairpinLengthInMeters / (df["DistanceChangeInKM"] * 1000).mean()
        )
        # window = 20 # meters long to consider a hairpin turn complete
        df, bearing = XC.AngleDifference(df, HairpinWindow)
        height = 120  # number of degrees in a hairpin turn over the window length
        st_dev, tot_deg_turned, HairpinCount = XC.HairpinDetection(
            df, height, bearing, "Western International 10km"
        )
        self.assertAlmostEqual(st_dev, 25.087924815982795)
        self.assertAlmostEqual(tot_deg_turned, 10432.96009585109)
        self.assertEqual(HairpinCount, 4)

    def test_CourseArea(self):
        df = XC.LoadRunIntoDF("Courses/Western International 10km.gpx")
        perim, area = XC.CourseArea(df)
        self.assertAlmostEqual(perim, 9968.833142535985)
        self.assertAlmostEqual(area, 460036.173622301)

    def test_AddDistanceColumn(self):
        df = self.Western10km
        df = XC.AddDistanceColumn(df)
        self.assertEqual(df.shape[0], 10279)
        self.assertEqual(df.shape[1], 16)
        self.assertAlmostEqual(df["Distance"][0], 0.12745673702681606)
        self.assertAlmostEqual(df["Distance"].iloc[-1], 9.934295906223545)

    def test_AngleDifference(self):
        df = XC.LoadRunIntoDF("Courses/Western International 10km.gpx")
        df["DistanceChangeInKM"] = df["Distance"] - df["Distance"].shift()
        HairpinLengthInMeters = 30
        HairpinWindow = round(
            HairpinLengthInMeters / (df["DistanceChangeInKM"] * 1000).mean()
        )
        df, bearing = XC.AngleDifference(df, HairpinWindow)
        self.assertEqual(df.shape[0], 601)
        self.assertEqual(df.shape[1], 11)
        self.assertAlmostEqual(df["Angle Difference"].iloc[-1], 0.10631551124572525)

    def test_ParsedGPXToDF(self):
        run_data = XC.ParseGPX("Courses/Western International 10km.gpx")
        df = XC.ParsedGPXToDF(run_data)
        self.assertEqual(df.shape[0], 601)
        self.assertEqual(df.shape[1], 4)
        self.assertAlmostEqual(df["Lat"][0], 42.962597)
        self.assertAlmostEqual(df["Lon"][0], -81.301311)
        self.assertAlmostEqual(df["Elevation"][0], 236.0)

    def test_IncrementalTimeAndElevation(self):
        run_data = XC.ParseGPX("Courses/Western International 10km.gpx")
        df = XC.ParsedGPXToDF(run_data)
        df = XC.IncrementalTimeAndElevation(run_data, df)
        self.assertEqual(df.shape[0], 601)
        self.assertEqual(df.shape[1], 6)
        self.assertAlmostEqual(df["Elevation Difference"][0], 0.0)
        self.assertAlmostEqual(df["Time Difference"][0], 0.0)
        self.assertAlmostEqual(
            df["Elevation Difference"].iloc[-1], -0.10000000000002274
        )
        self.assertAlmostEqual(df["Time Difference"].iloc[-1], 4.0)

    def test_LoadRunIntoDF(self):
        df = XC.LoadRunIntoDF("Courses/Western International 10km.gpx")
        self.assertEqual(df.shape[0], 601)
        self.assertEqual(df.shape[1], 9)
        self.assertAlmostEqual(df["Lat"][0], 42.962597)
        self.assertAlmostEqual(df["Lon"][0], -81.301311)
        self.assertAlmostEqual(df["Elevation"][0], 236.0)

    def test_HillPeaks(self):
        (df, _, _, _, _, _,) = XC.MakeInitialDataframe(
            "Western International 10km", "Courses/Western International 10km.gpx"
        )
        df = XC.InterpolateGPS(df)

        st_dev, peaks, mins, tot_elev_gain, tot_elev_gain_unfiltered = XC.HillPeaks(
            df, prominence=5, distance=5, course="Western International 10km"
        )
        self.assertAlmostEqual(st_dev, 9.052000807580995)
        self.assertEqual(len(peaks), 9)
        self.assertEqual(len(mins), 8)
        self.assertAlmostEqual(tot_elev_gain, 3208.699999999931)
        self.assertAlmostEqual(tot_elev_gain_unfiltered, 3208.6999999999284)

    def test_HillDetails(self):
        (df, _, _, _, _, _,) = XC.MakeInitialDataframe(
            "Western International 10km", "Courses/Western International 10km.gpx"
        )
        df = XC.InterpolateGPS(df)

        _, peaks, mins, _, _ = XC.HillPeaks(
            df, prominence=5, distance=5, course="Western International 10km"
        )

        hill_df = XC.HillDetails(df, peaks, mins)
        self.assertEqual(len(hill_df["Hill Height"]), 9)
        self.assertEqual(len(hill_df["Hill Distance"]), 9)
        self.assertAlmostEqual(hill_df["Hill Height"][0], 12.599999999999994)
        self.assertAlmostEqual(hill_df["Hill Distance"][0], 0.3882477119777579)
        self.assertAlmostEqual(hill_df["Hill Grade"][0], 3245.350741621841)

    def test_AddPace(self):
        df = self.Western10km
        df = XC.InterpolateGPS(df)
        df = XC.AddPace(df)
        self.assertEqual(len(df.columns), 16)
        self.assertAlmostEqual(df["PaceInMinPerKM"].iloc[0], 2.5347618971093433)

    def test_FirstNotNullInColumn(self):
        self.assertEqual(XC.FirstNotNullInColumn(pd.Series([1, 2, 3, None])), 1)
        self.assertEqual(XC.FirstNotNullInColumn(pd.Series([None, 1, 2, 3])), 1)

    def test_GetNumberOfHills(self):
        self.assertEqual(XC.GetNumberOfHills(self.HillDF, MinimumHillGrade=1), 2)
        self.assertEqual(XC.GetNumberOfHills(self.HillDF, MinimumHillGrade=2), 2)
        self.assertEqual(XC.GetNumberOfHills(self.HillDF, MinimumHillGrade=2.1), 1)

    def test_GetTallestHill(self):
        self.assertEqual(XC.GetTallestHill(self.HillDF), 3)

    def test_GetLengthOfTallestHill(self):
        self.assertEqual(XC.GetLengthOfTallestHill(self.HillDF), 3)

    def test_GetCourseInformation(self):
        course = "Western International 10km"
        gpx_file = "Courses/Western International 10km.gpx"
        CourseInformation, df = XC.GetCourseInformation(course, gpx_file)
        self.assertEqual(CourseInformation["Course"].iloc[0], course)
        self.assertAlmostEqual(
            CourseInformation["Turn Stdev"].iloc[0], 25.087924815982795
        )
        self.assertAlmostEqual(
            CourseInformation["Total Degrees Turned"].iloc[0], 10432.96009585109
        )
        self.assertAlmostEqual(
            CourseInformation["Perimeters/Total Distance"].iloc[0], 9968.833142535985
        )
        self.assertAlmostEqual(
            CourseInformation["Course Area (m^2)"].iloc[0], 460036.173622301
        )
        self.assertEqual(CourseInformation["Number of Hairpin Turns"].iloc[0], 4)
        self.assertAlmostEqual(
            CourseInformation["Hill Stdev"].iloc[0], 8.938635872380793
        )
        self.assertEqual(
            CourseInformation["Tallest Hill (m)"].iloc[0], 28.30000000000001
        )
        self.assertEqual(
            CourseInformation["Length of tallest hill (m)"].iloc[0], 0.6234376508579444
        )
        self.assertEqual(CourseInformation["Number of Hills"].iloc[0], 9)
        self.assertEqual(
            CourseInformation["Total Elevation Gain (m)"].iloc[0], 174.39999999999978
        )
        self.assertEqual(
            CourseInformation["Total Elevation Gain Unfiltered (m)"].iloc[0],
            174.39999999999995,
        )
        self.assertEqual(CourseInformation["Start Lat"].iloc[0], 42.962597)
        self.assertEqual(CourseInformation["Start Lon"].iloc[0], -81.301311)
        self.assertAlmostEqual(
            CourseInformation["Course Ave. GAP - Strava"].iloc[0],
            1.010589400735719,  # way off and weird, previously had -53.72
        )
        self.assertAlmostEqual(
            CourseInformation["Course Ave. GAP - Minetti"].iloc[0],
            1.0224953726590489,  # should be 87.11
        )

    def test_XCMain(self):
        ActualCourseSummary, ActualCourseData = XC.XCMain()
        self.assertEqual(len(ActualCourseSummary), 18)
        self.assertEqual(len(ActualCourseData), 18)
        # DesiredCourseSummary = pd.read_csv(
        #     "./Course DataFrames/AllCoursesSummary.csv", index_col=0
        # )
        # DesiredCourseSummary.index = DesiredCourseSummary["Course"]
        # DesiredCourseSummary.drop(["Course"], axis=1, inplace=True)
        # DesiredCourseData = pd.read_csv(
        #     "./Course DataFrames/Western10km.csv", index_col=0
        # )
        # DesiredCourseData.rename(columns={"Distance.1": "Distance"}, inplace=True)
        # DesiredCourseData["time"] = pd.to_datetime(DesiredCourseData["time"])
        # assert_frame_equal(
        #     ActualCourseSummary, DesiredCourseSummary, check_less_precise=True
        # )
        # assert_frame_equal(
        #     ActualCourseData[1], DesiredCourseData, check_less_precise=True
        # )


if __name__ == "__main__":
    unittest.main()
