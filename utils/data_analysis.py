import pandas as pd
from matplotlib import pyplot as plt


class DataAnalysis:
    def __init__(self, file_path: str):
        self.df = pd.read_csv(file_path)

    def class_distribution(self, target_col: str):
        class_value_counts = self.df[target_col].value_counts()
        print(class_value_counts, class_value_counts.__len__())
        s = self.df.groupby(target_col).indices.agg(pd.Series.nunique)
        pd.value_counts(s).plot(kind="bar")
        plt.show()
        return s

    def word_count_distribution(self):
        """
        word count distribution for the entire dataset
        :return:
        """
        pass

    def class_word_count_dist(self):
        """
        word count distribution for each class
        :return:
        """
        pass

    def sentiment_polarity_dist_per_class(self):
        """
        calculate sentiment distribution for each class
        :return:
        """
        pass

    def pos_dist_per_class(self):
        """
        parts of speech distribution for each class
        :return:
        """
        pass


if __name__ == '__main__':
    data_file_path = './../dataset/bmw_training_set.csv'
    explore_data_util = DataAnalysis(data_file_path)
    explore_data_util.class_distribution(target_col='Intent')
    pass
