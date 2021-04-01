import pandas as pd
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA
from random import random
import matplotlib.pyplot as plt
import pylab
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from statsmodels.tsa.ar_model import AutoReg, ar_select_order,AR
from random import random
from datetime import datetime
# import the necessary packages
from matplotlib import pyplot as plt


def dateparse (time_in_secs):
    return datetime.fromtimestamp(float(time_in_secs))


# out = x.truncate(before=datetime.datetime(2015,12,2,12,2,18))
class AnomalyDetection:

    @staticmethod
    def read_data():
        """

        :return:
        """
        raw_data = pd.read_csv("ec2_cpu_util.csv")
        return raw_data

    @staticmethod
    def moving_average(dataframe):
        """
        Moving Average Algorithm for predicting data-points.
        @:param dataframe: Pandas DF of the Data to be trained/tested.
        """
        train, test = train_test_split(dataframe, test_size=0.1, shuffle=False)
        predictions = test
        for index, value in test.iterrows():
            model = ARIMA(train.value.tolist(), order=(0, 0, 1))
            fit = model.fit()
            pred = fit.forecast()
            predictions['value'][index] = pred
            i = predictions.index[predictions.value == pred[0]]
            ts = predictions['timestamp'][i[0]]
            train.loc[len(train.index)] = [ts, pred[0]]

        dataframe.plot(x='timestamp', y='value', kind='scatter')
        plt.scatter(test.timestamp, test.value, c='blue')
        plt.scatter(predictions.timestamp, predictions.value, c='red')
        plt.show()

    @staticmethod
    def auto_regression(dataframe):
        """

        :return:
        """
        # fit model
        # X = dataframe.dropna().squeeze()
        train = dataframe[1:len(dataframe)-20]
        print("len(dataframe):"+str(len(dataframe)))
        # test= train
        test = dataframe.tail(20)



        # train, test = train_test_split(dataframe, test_size=0.20, shuffle=False)
        model = AutoReg(train.squeeze(), lags=[1, 11, 12])
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
        # pylab.show()
        #plot the two values
        plt.plot(compare_df)
        plt.show()
        # compare_df.plot()


        # r2 = r2_score(sales_data['stationary'].tail(12), predictions)

        # make prediction
        # yhat = model_fit.predict(len(data), len(data))
        # print(yhat)


if __name__ == '__main__':
    df = AnomalyDetection.read_data()
    AnomalyDetection.moving_average(df)
    AnomalyDetection.auto_regression(df)
