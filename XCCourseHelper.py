import pandas as pd
from scipy.signal import savgol_filter

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

def AboveOrEqualToThreshold(x, Threshold=0.05):
    if x >= Threshold:
        return x
    else:
        return 0
    
def myround(x, prec=1, base=0.5):
    return round(base * round(float(x)/base),prec)


def ElevationChange(Elevation):
    ElevationChange = Elevation.diff()
    return ElevationChange

def ElevationGain(ElevationChange):
    # ElevationGainPositive = ElevationChange.apply(positive_only).sum()
    ElevationGainAboveOrEqualToThreshold = ElevationChange.apply(AboveOrEqualToThreshold).sum()
    return ElevationGainAboveOrEqualToThreshold
