from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
import pandas as pd
import numpy as np
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import RandomOverSampler
from typing import Union
from keras.utils import to_categorical
from keras.metrics import TruePositives, TrueNegatives, Recall, Precision, FalseNegatives, FalsePositives, AUC

# TODO: use class to combine both the functions below to make it more efficient


def create_model(input_len: int, num_classes: int) -> Model:
    input1 = Input(shape=(input_len, ))

    layer1 = Dense(units=128, activation='relu')(input1)

    dropout_1 = Dropout(0.25)(layer1)

    layer2 = Dense(units=256, activation='relu')(dropout_1)

    dropout_2 = Dropout(0.25)(layer2)

    layer3 = Dense(units=num_classes, activation='softmax')(dropout_2)

    model = Model(inputs=[input1], outputs=[layer3])

    metrics = [
        Precision(name='precision'),
        Recall(name='recall'),
        AUC(name='auc')
    ]

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=metrics
    )
    model.summary()

    return model


def train(input_df: pd.DataFrame, model: Model, epochs: int, batch_size: int, num_classes: int) -> Model:
    """

    :param input_df:
    :param model:
    :param epochs:
    :param batch_size:
    :param num_classes:
    :return:
    """
    # print(input_df.head())
    x = input_df['Utterance'].values.tolist()
    y = input_df['Intent'].values.tolist()

    x = [np.asarray(i).flatten() for i in x]

    # oversampling to get more samples for classes with very little samples than other classes
    # to prevent overfitting
    combine_resample = RandomOverSampler()
    x, y = combine_resample.fit_resample(x, y)

    x = [np.asarray(i).astype('float32').reshape(1, -1) for i in x]
    x = np.asarray(x)
    # print(x.shape)

    y = to_categorical(y, num_classes=num_classes)
    y = [np.asarray(i).reshape(1, -1) for i in y]
    y = np.asarray(y)
    # print(y.shape)

    callbacks = list()
    early_stop = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True, min_delta=0.001, verbose=1)
    callbacks.append(early_stop)

    model.fit(
        x, y,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_split=0.15,
        shuffle=True,
        callbacks=callbacks,
    )
    return model


if __name__ == '__main__':
    df = pd.read_pickle('../utils/train.csv')
    model = create_model(input_len=768, num_classes=144)
    model = train(df, model, epochs=10, batch_size=8,  num_classes=144)

    pass
