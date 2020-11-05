import os
from keras.models import load_model, Model
from models import tf_classification
from sklearn.preprocessing import LabelEncoder
from utils import prepare_data
from utils import sent_to_vec
import numpy as np
from typing import Union
import pickle


class Main:
    def __init__(self, input_file: str, train_col: str, target_col: str, pre_trained_model: str,
                 encoder_file_path: Union[str, None]):
        """
        :param input_file: input data file path
        :param train_col: train column name
        :param target_col: target column name
        :param pre_trained_model: pre trained transformer based language model compatible with hugging face
        :param encoder_file_path: label encoder path
        """
        self.input_file = input_file
        self.train_col = train_col
        self.target_col = target_col
        self.pre_trained_model = pre_trained_model
        self.vectorizer = sent_to_vec.Sent2Vec(model_name=pre_trained_model)
        self.encoder_file_path = encoder_file_path
        if self.encoder_file_path:
            with open(self.encoder_file_path, 'rb') as fobj:
                self.encoder = pickle.load(fobj)
        else:
            self.encoder = LabelEncoder()
        self.num_classes = None
        self.model = None

    def run(self, model_save_path: Union[str, None], encoder_save_path: Union[str, None]) -> Model:
        """
        :param model_save_path:
        :return:
        """
        data_utils = prepare_data.PrepareData(file_path=self.input_file,
                                              train_col=self.train_col,
                                              target_col=self.target_col,
                                              pre_trained_model=self.pre_trained_model,
                                              sent_vectorizer=self.vectorizer,
                                              label_encoder=self.encoder)

        df, self.num_classes, self.encoder = data_utils.get_train_data()
        self.model = tf_classification.create_model(input_len=768, num_classes=self.num_classes)

        self.model = tf_classification.train(df, self.model, epochs=10, batch_size=8, num_classes=self.num_classes)
        if model_save_path:
            self.model.save(model_save_path)

        if encoder_save_path:
            with open(encoder_save_path, 'wb') as fobj:
                pickle.dump(self.encoder, fobj)

        return self.model

    def test_model(self, input_text: str, model_path: Union[str, None]):

        sent_embed = self.vectorizer.get_sent_embedding(input_text)
        if self.model:
            prediction = self.model.predict(sent_embed)
        else:
            self.model = load_model(model_path)
            prediction = self.model.predict(sent_embed)

        prediction = prediction.tolist()[0]
        index = np.argmax(prediction)

        predicted_class = self.encoder.inverse_transform([index])
        return predicted_class


if __name__ == '__main__':
    file_path = 'dataset/bmw_training_set.csv'
    main_func = Main(
        input_file=file_path,
        train_col='Utterance',
        target_col='Intent',
        pre_trained_model='bert-base-uncased',
        encoder_file_path='label_encoder.pkl'
    )
    # to train the model again uncomment the below line
    # main_func.run(None, None)
    while True:
        input_text = input("Input Text: ")
        predicted_class = main_func.test_model(input_text=input_text, model_path='model.h5')
        print(predicted_class)
    pass
