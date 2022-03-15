import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter


class GAP:
    def __init__(self, df):
        self.df = df

    def gradientCalculation(self):
        df = self.df
        df["Distance"] = df.index
        df["Gradient"] = (
            (df["Elevation"] - df["Elevation"].shift(1))
            / (df["Distance"] - df["Distance"].shift(1))
            / 1000
        ) * 100
        # # Max gradient is 100%
        # df["Gradient"] = df["Gradient"].apply(lambda x: x if x < 100 else 100)
        # # Min gradient is -100%
        # df["Gradient"] = df["Gradient"].apply(lambda x: x if x > -100 else -100)
        # df["Gradient"] = savgol_filter(df["Gradient"], 21, 2)
        df["Gradient"] = df["Gradient"].rolling(2).mean()
        self.df = df
        return self.df

    def stravaGapParameters(self):
        strava_df = pd.read_csv(
            "Gradient Adjustment/Strava Equal Heartrate.csv",
            header=None,
            names=["x", "y"],
            index_col=0,
        )
        strava_df = strava_df[strava_df.index > -30]  # cleaning noise
        xdata = strava_df.index
        ydata = strava_df["y"]

        def polynomialApproximation(x, a, b, c, d, e, f):
            return a * x ** 5 + b * x ** 4 + c * x ** 3 + d * x ** 2 + e * x + f

        self.popt, pcov = curve_fit(polynomialApproximation, xdata, ydata)

    # def polynomialApproximation(x, a, b, c, d, e, f):
    #     return a * x ** 5 + b * x ** 4 + c * x ** 3 + d * x ** 2 + e * x + f

    def minettiGapParameters(self):
        minetti_df = pd.read_csv(
            "Gradient Adjustment/Minetti-2002 Equal Energy Cost.csv",
            header=None,
            names=["x", "y"],
            index_col=0,
        )
        minetti_df = minetti_df[minetti_df.index > -30]  # cleaning noise
        xdata = minetti_df.index
        ydata = minetti_df["y"]

        def polynomialApproximation(x, a, b, c, d, e, f):
            return a * x ** 5 + b * x ** 4 + c * x ** 3 + d * x ** 2 + e * x + f

        self.popt, pcov = curve_fit(polynomialApproximation, xdata, ydata)

    def GapCalculation(self, name):
        f = (
            lambda x: self.popt[0] * x ** 5
            - self.popt[1] * x ** 4
            - self.popt[2] * x ** 3
            + self.popt[3] * x ** 2
            + self.popt[4] * x
            + self.popt[5]
        )
        self.df[name + " GAP"] = self.df["Gradient"].apply(f)

    def getStravaCourseGAP(self):
        self.gradientCalculation()
        self.stravaGapParameters()
        name = "Strava"
        self.GapCalculation(name)
        return self.df[name + " GAP"]

    def getStravaCourseGAPMean(self):
        return self.getStravaCourseGAP().mean()

    def getMinettiCourseGAP(self):
        self.gradientCalculation()
        self.minettiGapParameters()
        name = "Minetti"
        self.GapCalculation(name)
        return self.df[name + " GAP"]

    def getMinettiCourseGAPMean(self):
        return self.getMinettiCourseGAP().mean()

