import pandas as pd
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
        data = pd.read_csv('ec2_cpu_utilization_24ae8d.csv',parse_dates=['timestamp'],index_col=['timestamp'])
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
    data= AnomalyDetection.read_data()
    # print(data)
    AnomalyDetection.auto_regression(data)
