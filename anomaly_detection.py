import pandas as pd
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA
from random import random
import matplotlib.pyplot as plt


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


if __name__ == '__main__':
    df = AnomalyDetection.read_data()
    AnomalyDetection.moving_average(df)
    # data = [x + random() for x in range(1, 100)]
    # print(data)
