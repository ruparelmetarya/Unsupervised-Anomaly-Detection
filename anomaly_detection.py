import pandas as pd
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error
import pylab
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from statsmodels.tsa.ar_model import AutoReg, ar_select_order,AR
from random import random
from datetime import datetime
# import the necessary packages
from matplotlib import pyplot as plt

# create a difference transform of the dataset
def difference(dataset):
    diff = list()
    for i in range(1, len(dataset)):
        value = dataset[i] - dataset[i - 1]
        diff.append(value)
    return np.array(diff)

def getValue(diff,first):
    res = list()
    diff[0] = diff[0] + first
    for i in range(1, len(diff)-1):
        diff[i] = diff[i] +  diff[i-1]
    return np.array(diff)


# Make a prediction give regression coefficients and lag obs
def predict(coef, history):
    yhat = coef[0]
    for i in range(1, len(coef)):
        yhat += coef[i] * history[-i]
    return yhat
# out = x.truncate(before=datetime.datetime(2015,12,2,12,2,18))
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
        return data

    @staticmethod
    def moving_average():
        """

        :return:
        """

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
        train = dataframe[1:len(dataframe)-200]
        print("len(dataframe):"+str(len(dataframe)))
        test = dataframe.tail(200)



        model = AutoReg(train, lags=[1, 11, 12])
        model_fit = model.fit()
        print(model_fit.summary())
        # make predictions
        print(type(model_fit))

        predictions = model_fit.predict(start=len(train),
                                        end=len(train) + len(test)-1,
                                        dynamic=False)
        print(predictions)
        # create a comparison dataframe
        compare_df = pd.concat([test,
                                predictions], axis =1).rename(
                                    columns={'stationary': 'actual', 0:'predicted'})
        plt.plot(compare_df)
        plt.show()


if __name__ == '__main__':
    data= AnomalyDetection.read_data()
    # print(data)
    AnomalyDetection.auto_regression(data)
