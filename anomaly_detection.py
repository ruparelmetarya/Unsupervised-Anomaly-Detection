import pandas as pd


class AnomalyDetection:

    @staticmethod
    def read_data():
        """

        :return:
        """
        raw_data = pd.read_csv("/Users/Metarya/Downloads/trace_201708/container_usage.csv", header=None,
                               names=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'])
        del raw_data["2"]
        del raw_data["4"]
        del raw_data["5"]
        del raw_data["6"]
        del raw_data["7"]
        del raw_data["8"]
        del raw_data["9"]
        del raw_data["10"]
        del raw_data["11"]
        del raw_data["12"]
        raw_data.columns = ["timestamp", "cpu_util"]
        print(raw_data)

    @staticmethod
    def moving_average():
        """

        :return:
        """

    @staticmethod
    def auto_regression():
        """

        :return:
        """


if __name__ == '__main__':
    AnomalyDetection.read_data()
