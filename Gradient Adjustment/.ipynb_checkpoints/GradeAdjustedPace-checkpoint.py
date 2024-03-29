import pandas as pd
from scipy.optimize import curve_fit

class GAP:
    def __init__(self, df):
        self.df = df
    def gradientCalculation(self):
        df = self.df
        df['Gradient'] = ((df['Elevation']-df['Elevation'].shift(1))/(df['Distance']-df['Distance'].shift(1))/1000)*100
        df['Gradient'] = df['Gradient'].rolling(5).mean()
        self.df = df
    def stravaGapParameters(self):
        strava_df = pd.read_csv('Strava Equal Heartrate.csv', header=None, names=['x', 'y'], index_col=0)
        strava_df = strava_df[strava_df.index>-30] #cleaning noise
        xdata = strava_df.index
        ydata = strava_df['y']
        def polynomialApproximation(x, a, b, c, d, e, f):
            return a*x**5 + b*x**4 + c*x**3 + d*x**2 + e*x + f
        self.popt, pcov = curve_fit(polynomialApproximation, xdata, ydata)
    def stravaGapParameters(self):
        strava_df = pd.read_csv('Strava Equal Heartrate.csv', header=None, names=['x', 'y'], index_col=0)
        strava_df = strava_df[strava_df.index>-30] #cleaning noise
        xdata = strava_df.index
        ydata = strava_df['y']
        def polynomialApproximation(x, a, b, c, d, e, f):
            return a*x**5 + b*x**4 + c*x**3 + d*x**2 + e*x + f
        self.popt, pcov = curve_fit(polynomialApproximation, xdata, ydata)
    def minettiGapParameters(self):
        minetti_df = pd.read_csv('Minetti-2002 Equal Energy Cost.csv', header=None, names=['x', 'y'], index_col=0)
        minetti_df = minetti_df[strava_df.index>-30] #cleaning noise
        xdata = minetti_df.index
        ydata = minetti_df['y']
        def polynomialApproximation(x, a, b, c, d, e, f):
            return a*x**5 + b*x**4 + c*x**3 + d*x**2 + e*x + f
        self.popt, pcov = curve_fit(polynomialApproximation, xdata, ydata)
    def GapCalculation(self):
        f = lambda x: self.popt[0]*x**5 - self.popt[1]*x**4 - self.popt[2]*x**3 + self.popt[3]*x**2 + self.popt[4]*x + self.popt[5]
        self.df['GAP'] = self.df['Gradient'].apply(f)
    def getStravaCourseGAP(self):
        self.gradientCalculation()
        self.stravaGapParameters()
        self.GapCalculation()
        return self.df['GAP'].mean()
    def getMinettiCourseGAP(self):
        self.gradientCalculation()
        self.minettiGapParameters()
        self.GapCalculation()
        return self.df['GAP'].mean()