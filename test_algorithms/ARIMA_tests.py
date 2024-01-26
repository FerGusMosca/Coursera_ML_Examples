import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
import fsspec
import gcsfs
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from util.file_downloader import FileDownloader


class ARIMATests():


    _DOWNLOAD_URL="gs://cloud-training/ai4f/AAPL10Y.csv"
    _output_path = "./input/ARIMA_test.csv"


    def __init__(self):
        pass


    #region Private Static Methods


    @staticmethod
    def do_download():
        df = pd.read_csv(ARIMATests._DOWNLOAD_URL)
        df['date'] = pd.to_datetime(df['date'])
        df.sort_values('date', inplace=True)
        df.set_index('date', inplace=True)
        return df

    @staticmethod
    def aggregate_weekly_basis(df):
        df_week = df.resample('w').mean()
        df_week = df_week[['close']]
        df_week.head()
        return df_week

    @staticmethod
    def add_weekly_returns(df_week):
        df_week['weekly_ret'] = np.log(df_week['close']).diff()
        df_week.head()

        # drop null rows
        df_week.dropna(inplace=True)

        #En caso que lo queramos --> weekly plot
        #df_week.weekly_ret.plot(kind='line', figsize=(12, 6))

        udiff = df_week.drop(['close'], axis=1)
        udiff.head()

        return udiff

    @staticmethod
    def plot_rolling_mean_std_dev(udiff):
        rolmean = udiff.rolling(20).mean()
        rolstd = udiff.rolling(20).std()

        plt.figure(figsize=(12, 6))
        orig = plt.plot(udiff, color='blue', label='Original')
        mean = plt.plot(rolmean, color='red', label='Rolling Mean')
        std = plt.plot(rolstd, color='black', label='Rolling Std Deviation')
        plt.title('Rolling Mean & Standard Deviation')
        plt.legend(loc='best')
        plt.show(block=False)

    @staticmethod
    def perf_dickey_fuller_test(udiff):
        # Perform Dickey-Fuller test
        dftest = sm.tsa.adfuller(udiff.weekly_ret, autolag='AIC')
        dfoutput = pd.Series(dftest[0:4],
                             index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
        for key, value in dftest[4].items():
            dfoutput['Critical Value ({0})'.format(key)] = value

        print(dfoutput)


    @staticmethod
    #The ACF gives us a measure of how much each "y" value is correlated to the previous n "y" values prior.
    def acf_auto_corr_plot(udiff):
        fig, ax = plt.subplots(figsize=(12, 5))
        plot_acf(udiff.values, lags=10, ax=ax)
        plt.show()

    @staticmethod
    #he PACF is the partial correlation function gives us (a sample of) the amount of correlation between two "y" values
    # separated by n lags excluding the impact of all the "y" values in between them.
    def pacf_auto_corr_plot(udiff):
        fig, ax = plt.subplots(figsize=(12, 5))
        plot_pacf(udiff.values, lags=10, ax=ax)
        plt.show()


    @staticmethod
    def _build_ARIMA(udiff):
        # Notice that you have to use udiff - the differenced data rather than the original data.
        ar1 = ARIMA(udiff.values, order=(3, 0, 1)).fit()
        ar1.summary()

        # plt.figure(figsize=(12, 8))
        # plt.plot(udiff.values, color='blue')
        # preds = ar1.fittedvalues
        # plt.plot(preds, color='red')
        # plt.show()
        return ar1

    @staticmethod
    def run_forecast(udiff,ar1):
        steps=2
        forecast = ar1.forecast(steps=steps)

        plt.figure(figsize=(12, 8))
        plt.plot(udiff.values, color='blue')

        preds = ar1.fittedvalues
        plt.plot(preds, color='red')

        plt.plot(pd.DataFrame(np.array([preds[-1], forecast[0]]).T,
                              index=range(len(udiff.values) + 1, len(udiff.values) + 3)), color='green')
        plt.plot(pd.DataFrame(forecast, index=range(len(udiff.values) + 1, len(udiff.values) + 1 + steps)),
                 color='green')
        plt.title('Display the predictions with the ARIMA model')
        plt.show()


    #endregion



    #region Public Methods

    @staticmethod
    def test_stock_predictions():
        df= ARIMATests.do_download()

        df=ARIMATests.aggregate_weekly_basis(df)

        udiff=ARIMATests.add_weekly_returns(df)

        #ARIMATests.plot_rolling_mean_std_dev(udiff)

        #ARIMATests.perf_dickey_fuller_test(udiff)

        #ARIMATests.acf_auto_corr_plot(udiff)
        #ARIMATests.pacf_auto_corr_plot(udiff)
        ar1=ARIMATests._build_ARIMA(udiff)
        ARIMATests.run_forecast(udiff,ar1)
        pass

    #endregion