import pandas as pd
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense
import numpy as np
from keras.callbacks import TensorBoard, EarlyStopping
import datetime


def training_data(csv_path):
    df = pd.read_csv(csv_path)

    pclass_col = df['Pclass'].values
    age_col = df['Age'].values
    sibsp_col = df['SibSp'].values
    parch_col = df['Parch'].values
    df['Sex'] = df['Sex'].astype('category')
    df['Embarked'] = df['Embarked'].astype('category')
    df['Cabin_letter'] = df['Cabin'].str[0]
    df['Cabin_number'] = df['Cabin'].str[1:]

    # Replace empty strings with NaN
    df['Cabin_number'].replace('', np.nan, inplace=True)

    # Convert cabin number to numeric type (float)
    df['Cabin_number'] = pd.to_numeric(df['Cabin_number'], errors='coerce')

    # Fill missing values in cabin number with -1
    df['Cabin_number'].fillna(0, inplace=True)

    # Convert cabin letter to categorical type
    df['Cabin_letter'] = df['Cabin_letter'].astype('category')

    # Convert cabin number to int32
    df['Cabin_number'] = df['Cabin_number'].astype('int32')

    # Convert categorical columns to codes (int32)
    df['Cabin_letter'] = df['Cabin_letter'].cat.codes.astype('int32')

    pclass_tensor = tf.constant(pclass_col, dtype=tf.int16)
    age_tensor = tf.constant(age_col, dtype=tf.int16)
    sibsp_tensor = tf.constant(sibsp_col, dtype=tf.int16)
    parch_tensor = tf.constant(parch_col, dtype=tf.int16)
    sex_tensor = tf.constant(df['Sex'].cat.codes.values, dtype=tf.int16)
    embarked_tensor = tf.constant(df['Embarked'].cat.codes.values, dtype=tf.int16)
    cabin_letter_tensor = tf.constant(df['Cabin_letter'].values, dtype=tf.int16)
    cabin_number_tensor = tf.constant(df['Cabin_number'].values, dtype=tf.int16)

    data_train = (
        pclass_tensor,
        age_tensor,
        sibsp_tensor,
        parch_tensor,
        sex_tensor,
        embarked_tensor,
        cabin_letter_tensor,
        cabin_number_tensor)

    data_train_tensors = tf.stack(data_train)

    return data_train_tensors


def survived(path):
    df = pd.read_csv(path)
    survived_col = df['Survived'].values
    survived_tensor = tf.constant(survived_col, dtype=tf.int16)

    return survived_tensor


train_path = "\\path_to_file\\train.csv"


x_train, y_train = training_data(train_path), survived(train_path)
x_train = np.transpose(x_train)

model = Sequential()

model.add(Dense(128, activation='swish'))
model.add(Dense(128, activation='swish'))
model.add(Dense(256, activation='swish'))
model.add(Dense(256, activation='swish'))
model.add(Dense(1, activation='sigmoid'))

log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True)
early_stop = EarlyStopping(monitor='loss', patience=6)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

trained_model = model.fit(x_train, y_train, validation_split=0.03,epochs=50, batch_size=5,
                          callbacks=[[tensorboard_callback], [early_stop]])

model.save('TitanicAI_v9.2.h5')
