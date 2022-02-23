import pandas as pd
from scipy.signal import savgol_filter
import numpy as np
import time

def ChangeEvenToOdd(Number):
    if Number % 2 == 0:
        return Number + 1
    else:
        return Number

def KilometerToMeter(Kilometer):
    return Kilometer * 1000

def SavitzkyGolayFilter(df, WindowLengthInMeters, polyorder=3):
    try:
        KilometersInRace = df['Distance'].iloc[-1]
        MetersInRace = KilometerToMeter(KilometersInRace)
        WindowLength = MetersInRace / WindowLengthInMeters
        savgol_window_length = int(len(df)/WindowLength) #100m range for savgol filte
        savgol_window_length = ChangeEvenToOdd(savgol_window_length)
        df['Elevation SavGol'] = savgol_filter(df['Elevation'], savgol_window_length, polyorder)
        return df['Elevation SavGol']
    except: #TerreHaute doesn't work
        return df['Elevation']

def positive_only(x):
    if x > 0:
        return x
    else:
        return 0

def AboveOrEqualToThreshold(x, Threshold=0.04):
    if x >= Threshold:
        return x
    else:
        return 0
    
def myround(x, prec=1, base=0.5):
    return round(base * round(float(x)/base),prec)


def ElevationChange(Elevation):
    ElevationChange = Elevation.diff()
    return ElevationChange

def ElevationGainFromChange(ElevationChange):
    ElevationGainAboveOrEqualToThreshold = ElevationChange.apply(AboveOrEqualToThreshold).sum()
    return ElevationGainAboveOrEqualToThreshold

def GetElevationGain(Elevation):
    ElevationGain = ElevationGainFromChange(ElevationChange(Elevation))
    return ElevationGain

def ReadCourseDataFrame(CourseName):
    TestInputDirectory = './Course DataFrames/'
    CourseName =  CourseName + '.csv'
    try:
        data = pd.read_csv(TestInputDirectory + CourseName,
                           sep = ',',
                           header = 0)
        return data
    except IOError:
        print('cannot open file: ', CourseName)

def MakeInitialDataframe(course, gpx_file):
    df = load_run_to_df(gpx_file)

    df['DistanceChangeInKM'] = df['Distance']-df['Distance'].shift()
    
    HairpinLengthInMeters = 30
    HairpinWindow = round(HairpinLengthInMeters/(df['DistanceChangeInKM']*1000).mean())
    # window = 20 # meters long to consider a hairpin turn complete
    df, bearing = angle_diff(df, HairpinWindow)
    
    height = 120 #number of degrees in a hairpin turn over the window length
    turn_st_dev, tot_deg_turned, HairpinCount = hairpin_detection(df, height, bearing, course)
    
    perim, area = course_area(geod, df)
    
    df.set_index('Distance', inplace=True)

    return df, turn_st_dev, tot_deg_turned, HairpinCount, perim, area

def InterpolateGPS(df):
    FirstDistanceInKM = round(df.index[0],3)
    LastDistanceInKM = round(df.index[-1],3)
    df = df[~df.index.duplicated()] #some have duplicates that need to be removed
    df = df.reindex(df.index.union(np.linspace(FirstDistanceInKM, LastDistanceInKM,
                int((LastDistanceInKM-FirstDistanceInKM)*1000+1))))
    df.index = df.index.astype('float64')
    
    # Interpolate time
    #Cast date to seconds (also recast the NaT to Nan)
    df['seconds'] = [time.mktime(t.timetuple()) if t is not pd.NaT else float('nan') for t in df['time'] ]

    #Use the 'values'-argument to actually use the values of the index and not the spacing
    df['interpolated'] = df['seconds'].interpolate('values')
    #Cast the interpolated seconds back to datetime
    df['interpolated datetime']= pd.to_datetime(df['interpolated']-18000, unit='s', errors='coerce')

    #Clean up
    df['time'] = df['interpolated datetime']
    df.drop(columns=['interpolated datetime', 'interpolated', 'seconds'], inplace=True)

    # Interpolate everything else
    df = df.interpolate()
    return df