from random import random

import pandas as pd
import numpy as np
from math import sqrt
import pylab

import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_pacf
from pandas.plotting import autocorrelation_plot
import statsmodels.api as sm

from sklearn.model_selection import train_test_split

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.ar_model import AutoReg, ar_select_order, AR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


def getRecall(trueVals, preds):
    recall = recall_score(trueVals, preds)
    return recall


def getAccuracy(labels, preds):
    acc = accuracy_score(labels, preds)
    return acc


def getF1Score(trueVals, preds):
    f1Score = f1_score(trueVals, preds)
    return f1Score


def evaluate(test, predictions):
    rmse = sqrt(mean_squared_error(test, predictions))
    print('Test RMSE: %.3f' % rmse)
    mae = mean_absolute_error(test, predictions)
    print('Mean Absolute Error: %.3f' % mae)

    # evaluate forecasts
    test_dis = [1] * len(test)
    pred_dis = list()
    epsilon = 10 ** (-2)
    for i in range(len(test)):
        if (abs(test[i] - predictions[i]) < epsilon):
            pred_dis.append(1)
        else:
            pred_dis.append(0)

    print('Accuracy: ' + str(getAccuracy(test_dis, pred_dis)))
    print('F1Score: ' + str(getF1Score(test_dis, pred_dis)))
    print('Recall: ' + str(getRecall(test_dis, pred_dis)))
    print(confusion_matrix(test_dis, pred_dis))


# Make a prediction give regression coefficients and lag obs
def predict(coef, history):
    yhat = coef[0]
    for i in range(1, len(coef)):
        yhat += coef[i] * history[-i]
    return yhat


def dickey_fuller_obs(timeseries, window=12, cutoff=0.01):
    """

    :return:
    """
    # Determing rolling statistics
    rolmean = timeseries.rolling(window).mean()
    rolstd = timeseries.rolling(window).std()

    # Plot rolling statistics:
    fig = plt.figure(figsize=(12, 8))
    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()

    # Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC', maxlag=20)
    dfoutput = pd.Series(dftest[0:4],
                         index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    pvalue = dftest[1]
    if pvalue < cutoff:
        print('p-value = %.4f. The series is likely stationary.' % pvalue)
    else:
        print('p-value = %.4f. The series is likely non-stationary.' % pvalue)

    print(dfoutput)


class AnomalyDetection:
    def __init__(self,fileName):
        self.p = 3  # lag
        self.d = 0  # difference order
        self.q = 1  # size of moving average window
        self.fileName = fileName
        self.dataframe = self.read_data()

    def read_data(self):
        """

        :return:
        """
        data = pd.read_csv(self.fileName,
                           parse_dates=['timestamp'],
                           index_col=['timestamp'],
                           squeeze=True)
        # plt.rcParams['figure.figsize']=(20,10)
        # plt.style.use('ggplot')
        # data.plot()
        # plt.show()
        # print(data.value)
        # result = adfuller(data)
        # print('ADF Statistic: %f' % result[0])
        # print('p-value: %f' % result[1])
        # data.plot()
        # plt.show()
        return data

    def moving_average(self):
        """
        Moving Average Algorithm for predicting data-points.
        :param dataframe: Pandas DF of the Data to be trained/tested.
        """
        train, test = train_test_split(self.dataframe, test_size=0.1, shuffle=False)
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

        evaluate(test, predictions)
        # plot forecasts against actual outcomes
        plt.plot(test.index, test.values)
        plt.plot(test.index, predictions, color='red')
        plt.savefig("MA Prediction")

    def arima(self):
        """
        ARIMA Algorithm for predicting data-points.
        :param dataframe: Pandas DF of the Data to be trained/tested.
        """
        train, test = train_test_split(self.dataframe, test_size=0.1, shuffle=False)
        history = [x for x in train]
        predictions = list()
        # walk-forward validation
        for t in range(len(test)):
            arima_model = ARIMA(history, order=(self.p, self.d, self.q))
            model_fit = arima_model.fit()
            print(model_fit.summary)
            output = model_fit.forecast()
            yhat = output[0]
            predictions.append(yhat)
            obs = test[t]
            history.append(obs)
            print('predicted=%f, expected=%f' % (yhat, obs))

        evaluate(test, predictions)
        # plot forecasts against actual outcomes
        plt.plot(test.index, test.values)
        plt.plot(test.index, predictions, color='red')
        plt.savefig("ARIMA Prediction")

    def auto_regression(self):
        """

        :return:
        """
        dataframe = self.dataframe.sort_index()
        # plot_pacf(dataframe, lags=100)
        # plt.savefig("Lag Graph")

        train, test = train_test_split(dataframe, test_size=0.1, shuffle=False)
        predictions = list()
        model = AutoReg(train, lags=[1, 2, 61])
        model_fit = model.fit()
        # print(test)
        # print(type(test))
        # print(test.index[0])
        # print(model_fit.summary())

        # # walk-forward validation
        for t in range(len(test)):
            p = model_fit.predict(start=test.index[t],
                                  end=test.index[t],
                                  dynamic=False)
            # print(p)
            # print(p[0])
            predictions.append(p[0])
            # print('predicted=%f, expected=%f' % (p[0], test[t]))

        # plot forecasts against actual outcomes
        plt.plot(test.index, test.values)
        plt.plot(test.index, predictions, color='red')
        plt.show()
        # plt.savefig("AR Prediction")
        evaluate(test, predictions)

    def Sarimax(self):
        """

        :return:
        """
        dataframe = self.dataframe.sort_index()
        train, test = train_test_split(dataframe, test_size=0.1, shuffle=False)
        predictions = list()
        model = SARIMAX(dataframe, trend='c', order=(self.p,self.d,self.q))
        model_fit = model.fit()
        # # walk-forward validation
        for t in range(len(test)):
            p = model_fit.predict(start=test.index[t],
                                  end=test.index[t],
                                  dynamic=False)
            # print(p)
            # print(p[0])
            predictions.append(p[0])
            # print('predicted=%f, expected=%f' % (p[0], test[t]))

        # # plot forecasts against actual outcomes
        plt.plot(test.index, test.values)
        plt.plot(test.index, predictions, color='red')
        plt.show()
        # # plt.savefig("AR Prediction")
        evaluate(test, predictions)


if __name__ == '__main__':
    fileName = 'ec2_cpu_utilization_24ae8d.csv'
    anomalyDet = AnomalyDetection(fileName)
    anomalyDet.Sarimax()
    # anomalyDet.auto_regression(df)
    anomalyDet.arima()
    # df.columns = ['value']
    # AnomalyDetection.moving_average(df)
    # AnomalyDetection.arima(df)
    df = anomalyDet.dataframe
    dickey_fuller_obs(df)
    first_diff = df - df.shift(1)
    first_diff = first_diff.dropna(inplace=False)
    dickey_fuller_obs(first_diff)
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(df, lags=40, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(df, lags=40, ax=ax2)
    plt.show()
