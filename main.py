import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, Flatten
from tensorflow.keras.models import Model
import tensorflow as tf
import xgboost as xgb
from tensorflow.keras import backend as K

# Show all the collumns and rows
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# Hyperparameters
class BloodSugarPredictor(Model):
    def __init__(self):
        super(BloodSugarPredictor, self).__init__()

        # First LSTM layer with 64 units, returns sequences for next LSTM layer
        self.lstm1 = LSTM(units=128, return_sequences=True, activation='relu',
                          recurrent_activation='sigmoid', kernel_initializer='glorot_uniform',
                          recurrent_initializer='orthogonal', bias_initializer='zeros')

        self.dropout1 = Dropout(0.2)

        # Second LSTM layer with 32 units, does not return sequences
        self.lstm2 = LSTM(units=64, return_sequences=True, activation='relu',
                          recurrent_activation='sigmoid', kernel_initializer='glorot_uniform',
                          recurrent_initializer='orthogonal', bias_initializer='zeros')

        self.dropout2 = Dropout(0.2)

        self.lstm3 = LSTM(units=64, return_sequences=True, activation='relu',
                            recurrent_activation='sigmoid', kernel_initializer='glorot_uniform',
                            recurrent_initializer='orthogonal', bias_initializer='zeros')

        self.dropout3 = Dropout(0.2)

        # Third LSTM layer with 32 units, does not return sequences
        self.lstm4 = LSTM(units=32, activation='relu',
                            recurrent_activation='sigmoid', kernel_initializer='glorot_uniform',
                            recurrent_initializer='orthogonal', bias_initializer='zeros')

        # Dense layer for feature extraction
        self.fc1 = Dense(units=16, activation='relu')

        # Output layer with a single unit for regression output (blood sugar level prediction)
        self.output_layer = Dense(units=1, activation='linear')

    def call(self, inputs):
        # Pass through the first LSTM layer
        x = self.lstm1(inputs)
        x = self.dropout1(x)

        # Pass through the second LSTM layer
        x = self.lstm2(x)
        x = self.dropout2(x)
        # Pass through the third LSTM layer
        x = self.lstm3(x)
        x = self.dropout3(x)

        # Pass through the fourth LSTM layer
        x = self.lstm4(x)

        # Pass through the dense layer
        x = self.fc1(x)

        # Final output layer for prediction
        return self.output_layer(x)


def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_true - y_pred)))


def read_csv(file_path):
    if file_path.endswith('.csv'):
        data = pd.read_csv(file_path)
    else:
        data = pd.read_excel(file_path)
    return data


if __name__ == '__main__':

    # Read the prepared data
    X_train = pd.read_csv("Data/train_merge.csv")
    y_train = pd.read_csv("Data/y.csv")

    # drop bg+1:00 column from the X_train
    X_train = X_train.drop(columns=['bg+1:00'])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    # print(X_train.head())
    # print(y_train.head())
    # print(X_test.head())
    # print(y_test.head())

    # Convert to NumPy arrays
    X_train = X_train.values
    X_test = X_test.values

    # got the error: cannot reshape array of size 89574144 into shape (177024,60,64)
    X_train = X_train.reshape(-1, 1, 506)
    X_test = X_test.reshape(-1, 1, 506)

    # Instantiate and compile model
    model = BloodSugarPredictor()
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=[rmse, 'mae'])

    # Train the model
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

    # Example model summary (optional)
    model.summary()

    # Evaluate the model
    results = model.evaluate(X_test, y_test)
    print(f"Test RMSE: {results[1]}, Test MAE: {results[2]}")

    # Load the test data
    Real_X_test = pd.read_csv("Data/test_merge.csv")
    test_id = pd.read_csv("Data/test_id.csv")

    # Prepare the test data
    Real_X_test = Real_X_test.values
    Real_X_test = Real_X_test.reshape(-1, 1, 506)

    # Load the model
    # model = load_model("model.h5", custom_objects={'rmse': rmse})

    # Make predictions
    predictions = model.predict(Real_X_test)

    # Save the predictions by create a DataFrame with the test_id and the predictions
    predictions = pd.DataFrame(predictions, columns=['bg+1:00'])
    predictions['id'] = test_id['id']
    predictions.to_csv("Data/predictions.csv", index=False)

    print("Predictions saved successfully!")

    # save the model
    model.save("my_model", save_format="tf")