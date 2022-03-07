import pandas as pd
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
import numpy as np
import time
import gpxpy
from geopy import distance as gpyd
from math import sqrt, floor
from geographiclib.geodesic import Geodesic  # for heading

geod = Geodesic.WGS84  # define the WGS84 ellipsoid
import matplotlib.pyplot as plt
import statistics  # for standard deviation of turns

import matplotlib.pyplot as plt
from bokeh.plotting import figure
from bokeh.io import show

import copy
from scipy import stats  # for zscore filtering pace
import GradeAdjustedPace
import os


def ChangeEvenToOdd(Number):
    if Number % 2 == 0:
        return Number + 1
    else:
        return Number


def KilometerToMeter(Kilometer):
    return Kilometer * 1000


def SavitzkyGolayFilter(df, WindowLengthInMeters, polyorder=3):
    try:
        KilometersInRace = df["Distance"].iloc[-1]
        MetersInRace = KilometerToMeter(KilometersInRace)
        WindowLength = MetersInRace / WindowLengthInMeters
        savgol_window_length = int(
            len(df) / WindowLength
        )  # 100m range for savgol filte
        savgol_window_length = ChangeEvenToOdd(savgol_window_length)
        df["Elevation SavGol"] = savgol_filter(
            df["Elevation"], savgol_window_length, polyorder
        )
        return df["Elevation SavGol"]
    except:  # TerreHaute doesn't work
        return df["Elevation"]


def PositiveOnly(x):
    if x > 0:
        return x
    else:
        return 0


def AboveOrEqualToThreshold(x, Threshold=0.04):
    if x >= Threshold:
        return x
    else:
        return 0


def AdvancedRound(x, prec=1, base=0.5):
    return round(base * round(float(x) / base), prec)


def ElevationChange(Elevation):
    ElevationChange = Elevation.diff()
    return ElevationChange


def ElevationGainFromChange(ElevationChange):
    ElevationGainAboveOrEqualToThreshold = ElevationChange.apply(
        AboveOrEqualToThreshold
    ).sum()
    return ElevationGainAboveOrEqualToThreshold


def GetElevationGain(Elevation):
    ElevationGain = ElevationGainFromChange(ElevationChange(Elevation))
    return ElevationGain


def ReadCourseDataFrame(CourseName):
    TestInputDirectory = "./Course DataFrames/"
    CourseName = CourseName + ".csv"
    try:
        data = pd.read_csv(TestInputDirectory + CourseName, sep=",", header=0)
        return data
    except IOError:
        print("cannot open file: ", CourseName)


def PlotHairpins(df, bearing, peaks, CourseName):
    fig = plt.figure(figsize=(20, 8))
    plt.plot(df["Angle Difference"])
    plt.plot(bearing)
    plt.scatter(peaks, df["Angle Difference"].iloc[peaks])
    plt.title("Bearing and Bearing Changes - " + str(CourseName))
    plt.show()

    p = figure(
        title="Course Map with Hairpins - " + str(CourseName),
        x_axis_label="Lon",
        y_axis_label="Lat",
        plot_width=500,
        plot_height=500,
    )
    p.circle(df["Lon"], df["Lat"], size=0.5)
    p.circle(
        df["Lon"].iloc[peaks], df["Lat"].iloc[peaks], color="red", size=10, alpha=0.8
    )
    show(p)


def HairpinDetection(df, height, bearing, CourseName):
    peaks, _ = find_peaks(df["Angle Difference"], height=height, prominence=50)
    # PlotHairpins(df, bearing, peaks, CourseName)
    st_dev = statistics.stdev(df["Angle Difference"].dropna())
    tot_deg_turned = sum([abs(ele) for ele in df["Angle Difference"].dropna()])

    return st_dev, tot_deg_turned, len(peaks)


def CourseArea(df):
    p = geod.Polygon()
    for pnt in range(len(df)):
        if pnt == 0:
            continue
        p.AddPoint(df["Lat"][pnt], df["Lon"][pnt])

    num, perim, area = p.Compute()
    return perim, area


def AddDistanceColumn(df):
    tmp = [0]
    for i, item in enumerate(df.index):
        prevCoord = (df.iloc[i - 1]["Lat"], df.iloc[i - 1]["Lon"])
        currCoord = (df.iloc[i]["Lat"], df.iloc[i]["Lon"])
        tmp.append(tmp[-1] + gpyd.geodesic(prevCoord, currCoord).km)
    df["Distance"] = tmp[1:]
    return df


def AngleDifference(df, window):
    bearing = []

    for i in range(len(df["Lon"]) - 1):
        lon1 = df["Lon"].values[i]
        lat1 = df["Lat"].values[i]
        lon2 = df["Lon"].values[i + 1]
        lat2 = df["Lat"].values[i + 1]

        g = geod.Inverse(lat1, lon1, lat2, lon2)
        bearing.append((g["azi1"] + 360) % 360)

    bearing_diff = [bearing[i + 1] - bearing[i] for i in range(len(bearing) - 1)]
    angle_diff = [(bearing_diff[i] + 180) % 360 - 180 for i in range(len(bearing) - 1)]

    angle_df = pd.DataFrame(angle_diff).rolling(window=window, center=True).sum()
    angle_df = angle_df.abs()  # taking the absolute value so we get all the peask

    angle_df.loc[-1] = [np.nan]
    angle_df.loc[-2] = [np.nan]
    angle_df.index = angle_df.index + 3
    angle_df.sort_index(inplace=True)

    df["Angle Difference"] = angle_df[0]
    return df, bearing


def MakeInitialDataframe(CourseName, gpx_file):
    df = LoadRunIntoDF(gpx_file)

    df["DistanceChangeInKM"] = df["Distance"] - df["Distance"].shift()

    HairpinLengthInMeters = 30
    HairpinWindow = round(
        HairpinLengthInMeters / (df["DistanceChangeInKM"] * 1000).mean()
    )
    # window = 20 # meters long to consider a hairpin turn complete
    df, bearing = AngleDifference(df, HairpinWindow)

    height = 120  # number of degrees in a hairpin turn over the window length
    turn_st_dev, tot_deg_turned, HairpinCount = HairpinDetection(
        df, height, bearing, CourseName
    )

    perim, area = CourseArea(df)

    df.set_index("Distance", inplace=True)

    return df, turn_st_dev, tot_deg_turned, HairpinCount, perim, area


def InterpolateGPS(df):
    FirstDistanceInKM = round(df.index[0], 3)
    LastDistanceInKM = round(df.index[-1], 3)
    df = df[~df.index.duplicated()]  # some have duplicates that need to be removed
    df = df.reindex(
        df.index.union(
            np.linspace(
                FirstDistanceInKM,
                LastDistanceInKM,
                int((LastDistanceInKM - FirstDistanceInKM) * 1000 + 1),
            )
        )
    )
    df.index = df.index.astype("float64")

    # Interpolate time
    # Cast date to seconds (also recast the NaT to Nan)
    df["time"] = pd.to_datetime(df["time"])
    df["seconds"] = [
        time.mktime(t.timetuple()) if t is not pd.NaT else float("nan")
        for t in df["time"]
    ]

    # Use the 'values'-argument to actually use the values of the index and not the spacing
    df["interpolated"] = df["seconds"].interpolate("values")
    # Cast the interpolated seconds back to datetime
    df["interpolated datetime"] = pd.to_datetime(
        df["interpolated"] - 18000, unit="s", errors="coerce"
    )

    # Clean up
    df["time"] = df["interpolated datetime"]
    df.drop(columns=["interpolated datetime", "interpolated", "seconds"], inplace=True)

    # Interpolate everything else
    df = df.interpolate(method="ffill")
    return df


def ParseGPX(filename):
    with open(filename) as f:
        run_data = gpxpy.parse(f)
    f.closed
    return run_data


def ParsedGPXToDF(run_data):
    df_dict = {"time": [], "Lat": [], "Lon": [], "Elevation": []}
    df = pd.DataFrame(df_dict)

    for track in run_data.tracks:
        for segment in track.segments:
            for point in segment.points:
                df_newRow = pd.DataFrame(
                    [[point.time, point.latitude, point.longitude, point.elevation]],
                    columns=["time", "Lat", "Lon", "Elevation"],
                )
                df = df.append(df_newRow, ignore_index=True)
    return df


def IncrementalTimeAndElevation(run_data, df):
    alt_dif = [0]
    time_dif = [0]
    data = run_data.tracks[0].segments[0].points
    for index in range(len(data)):
        if index == 0:
            pass
        else:
            start = data[index - 1]
            stop = data[index]
            alt_d = start.elevation - stop.elevation
            alt_dif.append(alt_d)

            try:
                time_delta = (stop.time - start.time).total_seconds()
            except:
                # no time in data
                time_delta = 0

            time_dif.append(time_delta)

    df1 = pd.DataFrame()
    df1["Elevation Difference"] = alt_dif
    df1["Time Difference"] = time_dif
    df = pd.concat([df, df1], axis=1)
    return df


def LoadRunIntoDF(filename):
    run_data = ParseGPX(filename)
    df = ParsedGPXToDF(run_data)
    df["time"] = pd.to_datetime(df["time"])
    df = AddDistanceColumn(df)
    df["Elevation SavGol"] = SavitzkyGolayFilter(df, 1000)
    df["Elevation SavGol Difference"] = ElevationChange(df["Elevation SavGol"])
    df = IncrementalTimeAndElevation(run_data, df)
    return df


def plot_ele_max_min(df):
    # Plot results
    fig = plt.figure(figsize=(20, 8))
    plt.scatter(df["dis_hav_3d"], df["min"], c="r")
    plt.scatter(df["dis_hav_3d"], df["max"], c="g")
    plt.plot(df["dis_hav_3d"], df["Elevation"])
    plt.show()


# To be developed
def angle_between_points(lat, lon):
    # calculate angle between points
    # lat and lon are lists of points
    # returns angle in degrees
    # print(lat)
    # print(lon)
    bearing_tmp = atan2(
        sin(lon[1] - lon[0]) * cos(lat[1]),
        cos(lat[0]) * sin(lat[1]) - sin(lat[0]) * cos(lat[1]) * cos(lon[1] - lon[0]),
    )
    return degrees(bearing_tmp)


def angle_between(df):
    # Calculate bearing between each point
    df["Bearing"] = df.apply(
        lambda row: angle_between_points(row["Lat"], row["Lon"]), axis=1
    )
    return df


##


def HillPeaks(df, prominence, distance, course):
    peaks, _ = find_peaks(df["Elevation"], prominence=prominence, distance=distance)
    mins, _ = find_peaks(-df["Elevation"], prominence=prominence, distance=distance)

    # PlotElevation(df, course, peaks, mins)

    st_dev = statistics.stdev(df["Elevation"].dropna())

    pos_only = list(map(PositiveOnly, df["Elevation Difference"]))
    tot_elev_gain = sum(
        list(map(lambda x: AdvancedRound(x, prec=2, base=0.05), pos_only))
    )
    tot_elev_gain_unfiltered = sum(pos_only)

    savgol_pos_only = list(map(PositiveOnly, df["Elevation SavGol Difference"]))
    tot_elev_gain_savgol = sum(savgol_pos_only)
    # PlotElevationDifference(df)

    return st_dev, peaks, mins, tot_elev_gain, tot_elev_gain_unfiltered


def PlotElevationDifference(df):
    fig = plt.figure(figsize=(20, 8))
    plt.scatter(df.index, df["Elevation Difference"])
    plt.plot(df.index, df["Elevation SavGol Difference"], color="red")
    plt.show()


def PlotElevation(df, course, peaks, mins):
    #     plot it
    fig = plt.figure(figsize=(20, 8))
    #     plt.rcParams.update({'font.size': 22})
    plt.plot(df["Elevation"])
    plt.plot(df.index, df["Elevation SavGol"], color="green")
    plt.scatter(df.index[peaks], df["Elevation"].iloc[peaks], color="red")
    plt.scatter(df.index[mins], df["Elevation"].iloc[mins], color="blue")
    plt.title("Elevation and Peaks - " + str(course))
    plt.show()

    # plot them on the course
    # bokeh
    p = figure(
        title="Course Map with Hill Peaks - " + str(course),
        x_axis_label="Lon",
        y_axis_label="Lat",
        plot_width=500,
        plot_height=500,
    )
    # Add a circle glyph to the figure p
    p.circle(df["Lon"], df["Lat"], size=0.5)
    p.circle(
        df["Lon"].iloc[peaks], df["Lat"].iloc[peaks], color="red", size=10, alpha=0.8
    )
    p.circle(
        df["Lon"].iloc[mins], df["Lat"].iloc[mins], color="blue", size=10, alpha=0.8
    )
    # Display the plot
    show(p)


def HillDetails(df, peaks, mins):
    # if the first point is a peak then set the start as the min
    if len(peaks) > 0 and peaks[0] < mins[0]:
        mins = np.insert(mins, 0, 0)

    hill_climbs = []
    hill_lengths = []

    for i in range(len(peaks)):  # loop through each max location
        try:  # error handling for weird numbered hills
            min_val = df["Elevation"].iloc[mins[i]]  # get this hill's min elevation
            max_val = df["Elevation"].iloc[peaks[i]]  # get this hill's max elevation
            hill_climb = max_val - min_val  # get the hill climb amount

            hill_start = df.index[mins[i]]  # get this hill's start point
            hill_end = df.index[peaks[i]]  # get this hill's end point
            hill_length = hill_end - hill_start

            hill_climbs.append(hill_climb)
            hill_lengths.append(hill_length)
        except:
            print("one hill error")

    hill_grade = np.divide(hill_climbs, hill_lengths) * 100
    hill_df = pd.DataFrame(
        {
            "Hill Height": hill_climbs,
            "Hill Distance": hill_lengths,
            "Hill Grade": hill_grade,
        }
    )

    return hill_df


def AddPace(df):

    # drop when distance change is zero for unknown reasons
    # df = df[df["DistanceChangeInKM"] != 0]

    pace_hist = ((df["Time Difference"] / 60) / df["DistanceChangeInKM"]).values
    df["PaceInMinPerKM"] = pace_hist

    # remove spikes (if a value is more than spike_tolerace times larger than the mean of the surrounding values)
    spike_tolerance = 1.3
    pace_hist_nospikes = copy.copy(pace_hist)
    for i in range(len(pace_hist_nospikes)):
        if i > 1 and i < (len(pace_hist_nospikes) - 2):
            if pace_hist_nospikes[i] > spike_tolerance * np.mean(
                [
                    pace_hist_nospikes[i - 2],
                    pace_hist_nospikes[i - 1],
                    pace_hist_nospikes[i + 1],
                    pace_hist_nospikes[i + 2],
                ]
            ):
                pace_hist_nospikes[i] = np.mean(
                    [pace_hist_nospikes[i - 2], pace_hist_nospikes[i + 2]]
                )

    # running average of pace (without spikes), N is width of average
    N = 23  # N must be 2n+1
    pace_hist_binned = np.convolve(pace_hist_nospikes, np.ones((N,)) / N, mode="valid")

    # Add Pace
    df["pace"] = np.nan
    df["pace"].iloc[int((N - 1) / 2) : -int((N - 1) / 2)] = pace_hist_binned

    # drop outlier in pace (sometimes watch seems to be stopped early or late)
    df = df.dropna()[(np.abs(stats.zscore(df.dropna()["pace"])) < 3)]

    # PlotPace(df)

    return df


## Missing tests, above should be cleaned up. AddPace is really slow


def PlotPace(df):
    # Plot original pace trace in grey and spike-removed, averaged trace in red
    fig = plt.figure(figsize=(20, 8))
    plt.plot(df.index, df["pace"], "r")  # rolling average plot
    plt.title("Pace")
    plt.ylabel("pace (min per km)")
    plt.xlabel("distance (km)")
    plt.show()


def PlotElevation(course, df):
    fig = plt.figure(figsize=(10, 10))
    plt.title(course + " - Elevation")
    plt.scatter(x=df["Lat"], y=df["Lon"], c=df["Elevation"], cmap=plt.cm.inferno)
    plt.ylabel("Lon")
    plt.xlabel("Lat")
    buffer = 0.0001
    plt.xlim(df["Lat"].min() - buffer, df["Lat"].max() + buffer)
    plt.ylim(df["Lon"].min() - buffer, df["Lon"].max() + buffer)

    cbar = plt.colorbar()
    cbar.set_label("Elevation (m)")

    plt.show()


def PlotElevationAndPace(course, df):
    fig = plt.figure(figsize=(10, 10))
    plt.title(course + " - Elevation and Pace")
    # sc = plt.scatter(x=df['Lat'], y=df['Lon'], c=df['pace'], cmap=plt.cm.inferno, s = scaler.transform(df[['Elevation']])*50)
    h_min = df["Elevation"].min()
    h_const = 10  # constant to adjust the heights
    f = lambda a: a - h_min + h_const
    g = lambda s: s + h_min - h_const
    sc = plt.scatter(
        x=df["Lat"],
        y=df["Lon"],
        c=df["pace"],
        cmap=plt.cm.inferno,
        s=f(df[["Elevation"]].values),
    )
    # sc = plt.scatter(x=df['Lat'], y=df['Lon'], c=df['pace'], cmap=plt.cm.inferno, s = df['Elevation'])
    plt.ylabel("Lon")
    plt.xlabel("Lat")
    buffer = 0.0001
    plt.xlim(df["Lat"].min() - buffer, df["Lat"].max() + buffer)
    plt.ylim(df["Lon"].min() - buffer, df["Lon"].max() + buffer)
    cbar = plt.colorbar()
    cbar.set_label("Pace (min)")

    plt.legend(*sc.legend_elements("sizes", num=5, func=g), title="Elevation (m)")

    plt.show()


def FirstNotNullInColumn(Column):
    return Column.loc[~Column.isnull()].iloc[0]


def GetNumberOfHills(hill_df, MinimumHillGrade=1):
    hill_df.drop(hill_df[hill_df["Hill Grade"] < MinimumHillGrade].index, inplace=True)
    hill_df.reset_index(drop=True, inplace=True)
    num_hills = len(hill_df)
    return num_hills


def GetTallestHill(hill_df):
    return hill_df["Hill Height"].max()


def GetLengthOfTallestHill(hill_df):
    if len(hill_df) > 0:
        lengthoftallesthill = hill_df[
            hill_df["Hill Height"] == max(hill_df["Hill Height"])
        ]["Hill Distance"].min()
    else:
        lengthoftallesthill = np.nan
    return lengthoftallesthill


def GetCourseInformation(course, gpx_file):
    df, turn_st_dev, tot_deg_turned, HairpinCount, perim, area = MakeInitialDataframe(
        course, gpx_file
    )

    # Interpolate
    df = InterpolateGPS(df)

    if df["time"].isnull().all() == True:
        print("no time")
    else:
        df = AddPace(df)

    hill_min_height = 5
    length_of_hill = 5
    hill_st_dev, peaks, mins, tot_elev_gain, tot_elev_gain_unfiltered = HillPeaks(
        df, hill_min_height, length_of_hill, course
    )

    hill_df = HillDetails(df, peaks, mins)
    NumberOfHills = GetNumberOfHills(hill_df)
    TallestHill = GetTallestHill(hill_df)
    LengthOfTallestHill = GetLengthOfTallestHill(hill_df)

    StartingLatitude = FirstNotNullInColumn(df["Lat"])
    StartingLongitude = FirstNotNullInColumn(df["Lon"])

    # PlotElevation(course, df)

    # if df["time"].isnull().all() == True:
    #     print("no time")
    # else:
    #     PlotElevationAndPace(course, df)

    GAP = GradeAdjustedPace.GAP(df)
    courseAverageGradeAdjustedPaceStrava = GAP.getStravaCourseGAP()
    courseAverageGradeAdjustedPaceMinetti = GAP.getMinettiCourseGAP()

    CourseInformation = pd.DataFrame(
        {
            "Course": course,
            "Turn Stdev": turn_st_dev,
            "Total Degrees Turned": tot_deg_turned,
            "Perimeters/Total Distance": perim,
            "Course Area (m^2)": area,
            "Number of Hairpin Turns": HairpinCount,
            "Hill Stdev": hill_st_dev,
            "Tallest Hill (m)": TallestHill,
            "Length of tallest hill (m)": LengthOfTallestHill,
            "Number of Hills": NumberOfHills,
            "Total Elevation Gain (m)": tot_elev_gain,
            "Total Elevation Gain Unfiltered (m)": tot_elev_gain_unfiltered,
            "Start Lat": StartingLatitude,
            "Start Lon": StartingLongitude,
            "Course Ave. GAP - Strava": courseAverageGradeAdjustedPaceStrava,
            "Course Ave. GAP - Minetti": courseAverageGradeAdjustedPaceMinetti,
        },
        index=[0],
    )
    return CourseInformation, df


def XCMain():
    directory = "Courses"
    dfs = []
    CourseInformationList = []

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        print(filename)
        if filename.endswith(".gpx"):
            FilePath = os.path.join(directory, filename)
            # checking if it is a file
            if os.path.isfile(FilePath):
                CourseName = filename[0:-4]
                CourseInformation, df = GetCourseInformation(CourseName, FilePath)
                CourseInformationList.append(CourseInformation)
                dfs.append(df)

    CourseInformationDF = pd.concat(CourseInformationList, ignore_index=True)
    return CourseInformationDF, dfs
