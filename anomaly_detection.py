from math import sqrt

import pandas as pd
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA
from random import random
import matplotlib.pyplot as plt
import pylab
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from statsmodels.tsa.ar_model import AutoReg, ar_select_order, AR
from random import random
from datetime import datetime
# import the necessary packages
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import adfuller
from pandas.plotting import autocorrelation_plot


def dateparse(time_in_secs):
    return datetime.fromtimestamp(float(time_in_secs))


from sklearn.metrics import r2_score
from statsmodels.tsa.ar_model import AutoReg, ar_select_order, AR


# create a difference transform of the dataset
def difference(dataset):
    diff = list()
    for i in range(1, len(dataset)):
        value = dataset[i] - dataset[i - 1]
        diff.append(value)
    return np.array(diff)


def getValue(diff, first):
    res = list()
    diff[0] = diff[0] + first
    for i in range(1, len(diff) - 1):
        diff[i] = diff[i] + diff[i - 1]
    return np.array(diff)


# Make a prediction give regression coefficients and lag obs
def predict(coef, history):
    yhat = coef[0]
    for i in range(1, len(coef)):
        yhat += coef[i] * history[-i]
    return yhat


class AnomalyDetection:

    @staticmethod
    def read_data():
        """

        :return:
        """
        data = pd.read_csv('ec2_cpu_utilization_24ae8d.csv',
                           parse_dates=['timestamp'],
                           index_col=['timestamp'],
                           squeeze=True)
        # plt.rcParams['figure.figsize']=(20,10)
        # plt.style.use('ggplot')
        # data.plot()
        # plt.show()
        # print(data.value)
        result = adfuller(data)
        print('ADF Statistic: %f' % result[0])
        print('p-value: %f' % result[1])
        data.plot()
        plt.show()
        return data

    @staticmethod
    def moving_average(dataframe):
        """
        Moving Average Algorithm for predicting data-points.
        @:param dataframe: Pandas DF of the Data to be trained/tested.
        """
        train, test = train_test_split(dataframe, test_size=0.1, shuffle=False)
        history = [x for x in train]
        predictions = list()
        # walk-forward validation
        for t in range(len(test)):
            model = ARIMA(history, order=(0, 0, 1))
            model_fit = model.fit()
            print(model_fit.summary)
            output = model_fit.forecast()
            yhat = output[0]
            predictions.append(yhat)
            obs = test[t]
            history.append(obs)
            print('predicted=%f, expected=%f' % (yhat, obs))
        # evaluate forecasts
        rmse = sqrt(mean_squared_error(test, predictions))
        print('Test RMSE: %.3f' % rmse)
        # plot forecasts against actual outcomes
        plt.plot(test.index, test.values)
        plt.plot(test.index, predictions, color='red')
        plt.show()

    # @staticmethod
    # def ar2(series):
    #     X = difference(series.values)
    #     size = int(len(X) * 0.66)
    #     train, test = X[0:size], X[size:]
    #     t2 = series.values[size:]
    #     # train autoregression
    #     window = 6
    #     model = AutoReg(train, lags=6)
    #     model_fit = model.fit()
    #     coef = model_fit.params
    #     # walk forward over time steps in test
    #     history = [train[i] for i in range(len(train))]
    #     predictions = list()
    #     for t in range(len(test)):
    #         yhat = predict(coef, history)
    #         obs = test[t]
    #         predictions.append(yhat)
    #         history.append(obs)
    #     rmse = sqrt(mean_squared_error(test, predictions))
    #     print('Test RMSE: %.3f' % rmse)
    #     # plot
    #     # plt.plot(test)
    #     # test2 = getValue(test)
    #     plt.plot(t2)
    #     print(series.values[size-3:size+3])
    #     plt.plot(predictions, color='red')
    #     plt.show()

    @staticmethod
    def auto_regression(dataframe):
        """

        :return:
        """


if __name__ == '__main__':
    df = AnomalyDetection.read_data()
    # AnomalyDetection.moving_average(df)
    # AnomalyDetection.auto_regression(df)
