import pandas as pd
import os
import swifter
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import pickle
from typing import Union

from utils.sent_to_vec import Sent2Vec


class PrepareData:
    def __init__(self, file_path: str, train_col: str, target_col: str, sent_vectorizer,
                 label_encoder: LabelEncoder, pre_trained_model: str = 'bert-base-uncased'):
        self.file_path = file_path
        self.train_col = train_col
        self.target_col = target_col
        self.vectorizer = sent_vectorizer
        self.encoder = label_encoder

    def _read_data(self) -> pd.DataFrame:

        df = pd.read_csv(self.file_path)

        df.info()

        return df

    def _convert_sent_to_vec(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.train_col] = df[self.train_col].swifter.apply(self.vectorizer.get_sent_embedding)

        return df

    def _convert_label_to_vec(self, df: pd.DataFrame) -> [pd.DataFrame, int]:
        num_classes = df[self.target_col].unique().__len__()
        df[self.target_col] = self.encoder.fit_transform(df[self.target_col])

        return df, num_classes, self.encoder

    def get_train_data(self) -> [pd.DataFrame, int]:
        df = self._read_data()
        df, num_classes, encoder = self._convert_label_to_vec(df)
        df = self._convert_sent_to_vec(df)

        return df, num_classes, encoder


if __name__ == '__main__':
    train_file_path = './../dataset/bmw_training_set.csv'
    # temp_df = pd.read_csv(train_file_path)
    # total_classes = temp_df['Intent'].unique().__len__()
    # print(total_classes)
    # exit(0)
    vectorizer = Sent2Vec()
    data_util = PrepareData(file_path=train_file_path,
                            train_col='Utterance',
                            target_col='Intent',
                            pre_trained_model='bert-base-uncased')

    train_df, total_classes = data_util.get_train_data()
    train_df.info()
    train_df.to_pickle('train.csv')
    pass
