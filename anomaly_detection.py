import pandas as pd


class AnomalyDetection:

    @staticmethod
    def read_data():
        """

        :return:
        """
        raw_data = pd.read_csv('value.txt')
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
