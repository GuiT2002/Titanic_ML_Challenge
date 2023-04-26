import tensorflow as tf
from keras.models import load_model
import pandas as pd
import numpy as np
import csv

model = load_model('TitanicAI_v9.2.h5')

test_path = "C:\\Users\\huilh\\OneDrive\\Área de Trabalho\\AI Training Models\\Kaggle Competitions\\Titanic - " \
            "Machine Learning from Disaster\\test.csv"


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


data = np.transpose(training_data(test_path))


predictions = model.predict(data)

# Open a new CSV file for writing

with open('predictions20.csv', mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)

    # Write the header row
    writer.writerow(['PassengerId', 'Survived'])

    # Write the predictions for each passenger to the file
    for i in range(len(predictions)):
        passenger_id = i + 892  # passenger IDs start at 892 in the test data
        survived = int(predictions[i][0]+0.5)  # round the prediction to 0 or 1
        writer.writerow([passenger_id, survived])

print(predictions)

