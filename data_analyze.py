import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler


def set_display():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 1000)
    warnings.filterwarnings('ignore')


def data_load():
    df_train = pd.read_csv("Data/train.csv")
    df_test = pd.read_csv("Data/test.csv")
    df_filled_train = pd.read_csv("Data/filled_data_train.csv")
    df_filled_test = pd.read_csv("Data/filled_data_test.csv")
    df_filled_train = pd.DataFrame(df_filled_train)
    df_filled_test = pd.DataFrame(df_filled_test)

    return df_train, df_test, df_filled_train, df_filled_test


'''def data_filling_with_ai(data):
    print("Data filling process completed using AI model.")
    numerical_columns = [col for col in data.columns if
                         col.startswith(('bg-', 'insulin-', 'carbs-', 'hr-', 'steps-', 'cals-'))]

    numerical_data = data[numerical_columns]
    print("count of missing values in numerical columns: ", numerical_data.isnull().sum())
    data_filled = numerical_data.fillna(numerical_data.mean())
    input_dim = data_filled.shape[1]
    latent_dim = 16
    timesteps = 10
    features = data_filled.shape[1]

    inputs = Input(shape=(timesteps, features))
    encoder = LSTM(64, return_sequences=True)(inputs)
    encoder = LSTM(latent_dim, return_sequences=False)(encoder)
    latent = encoder

    decoder = RepeatVector(timesteps)(latent)
    decoder = LSTM(64, return_sequences=True)(decoder)
    outputs = TimeDistributed(Dense(features))(decoder)

    vae_lstm = Model(inputs, outputs)
    vae_lstm.compile(optimizer='adam', loss='mse')

    n_samples = len(data_filled)
    valid_samples = (n_samples // timesteps) * timesteps
    data_filled_trimmed = data_filled.iloc[:valid_samples]
    data_series = data_filled_trimmed.values.reshape(-1, timesteps, features)
    vae_lstm.fit(data_series, data_series, epochs=100, batch_size=32)

    New_Series = vae_lstm.predict(data_series)
    New_pd = pd.DataFrame(New_Series.reshape(-1, features), columns=numerical_columns)

    print(New_pd.head())

    New_pd.to_csv("Data/filled_data_test.csv", index=False)

    return New_pd'''


def preprocess_data(train, test, df_filled_train, df_filled_test, target_column="bg+1:00"):
    train_first_3_column = train.iloc[:, :3]
    test_first_3_column = test.iloc[:, :3]
    y_backup = train[target_column]

    train_first_3_column['hour'] = pd.to_datetime(train_first_3_column['time'], format='%H:%M:%S').dt.hour
    train_first_3_column['minute'] = pd.to_datetime(train_first_3_column['time'], format='%H:%M:%S').dt.minute

    test_first_3_column['hour'] = pd.to_datetime(test_first_3_column['time'], format='%H:%M:%S').dt.hour
    test_first_3_column['minute'] = pd.to_datetime(test_first_3_column['time'], format='%H:%M:%S').dt.minute

    train = train.iloc[:, 3:-1]
    test = test.iloc[:, 3:]

    numerical_columns_train = [col for col in train.columns if
                               col.startswith(('bg-', 'insulin-', 'carbs-', 'hr-', 'steps-', 'cals-'))]
    categorical_columns_train = [col for col in train.columns if col not in numerical_columns_train]

    numerical_columns_test = [col for col in test.columns if
                              col.startswith(('bg-', 'insulin-', 'carbs-', 'hr-', 'steps-', 'cals-'))]
    categorical_columns_test = [col for col in test.columns if col not in numerical_columns_test]

    for col in categorical_columns_train:
        train[categorical_columns_train] = train[categorical_columns_train].fillna(0)
        train[col] = train[col].astype('category').cat.codes

    for col in categorical_columns_test:
        test[categorical_columns_test] = test[categorical_columns_test].fillna(0)
        test[col] = test[col].astype('category').cat.codes

    train_without_original = train.drop(columns=[col for col in train.columns if
                                                 col.startswith(
                                                     ('bg-', 'insulin-', 'carbs-', 'hr-', 'steps-', 'cals-'))])
    test_without_original = test.drop(columns=[col for col in test.columns if
                                               col.startswith(('bg-', 'insulin-', 'carbs-', 'hr-', 'steps-', 'cals-'))])

    train_full = pd.concat([train_first_3_column, df_filled_train, train_without_original, y_backup], axis=1)
    test_full = pd.concat([test_first_3_column, df_filled_test, test_without_original], axis=1)

    test_id_column = test_full["id"]

    train_full = train_full.drop(columns=["id", "p_num", "time"])
    test_full = test_full.drop(columns=["id", "p_num", "time"])

    train_full = train_full.fillna(0)
    test_full = test_full.fillna(0)
    y_backup = y_backup.fillna(0)

    return train_full, test_full, y_backup, test_id_column


if __name__ == "__main__":
    train, test, filled_train, filled_test = data_load()
    train_merge, test_merge, y, test_id = preprocess_data(train, test, filled_train, filled_test)

    # save train and test data as csv files
    train_merge.to_csv("Data/train_merge.csv", index=False)
    test_merge.to_csv("Data/test_merge.csv", index=False)
    y.to_csv("Data/y.csv", index=False)
    test_id.to_csv("Data/test_id.csv", index=False)

    '''X = train_merge.drop(columns=["bg+1:00"])
    y_data = train_merge["bg+1:00"]
    X_train, X_test, y_train, y_test = train_test_split(X, y_data, test_size=0.2, random_state=42)

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # LSTM için veri şekillendirme
    X_train_reshaped = np.reshape(X_train_scaled, (X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    X_test_reshaped = np.reshape(X_test_scaled, (X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

    # LSTM modelini oluşturma (optimize edilmiş)
    model = Sequential()
    model.add(LSTM(units=64, return_sequences=True, input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])))
    model.add(LSTM(units=32))
    model.add(Dense(1))

    # Modeli derleme
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Erken durdurma tanımlama
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Modeli eğitme (optimize edilmiş epoch ve batch_size)
    history = model.fit(X_train_reshaped, y_train, epochs=100, batch_size=32,
                        validation_data=(X_test_reshaped, y_test),
                        callbacks=[early_stopping], verbose=1)

    # Eğitim ve test tahminleri
    train_predictions = model.predict(X_train_reshaped)
    test_predictions = model.predict(X_test_reshaped)

    # Eğitim ve test hataları
    train_score = mean_squared_error(y_train, train_predictions)
    test_score = mean_squared_error(y_test, test_predictions)

    print("Train MSE:", train_score)
    print("Test MSE:", test_score)

    # Test verisi tahminleri
    test_merge_scaled = scaler.transform(test_merge)
    test_merge_reshaped = np.reshape(test_merge_scaled, (test_merge_scaled.shape[0], 1, test_merge_scaled.shape[1]))

    predictions = model.predict(test_merge_reshaped).flatten()

    # Tersine ölçeklendirme için dummy veri oluşturma
    dummy_data = np.zeros((predictions.shape[0], scaler.min_.shape[0]))
    dummy_data[:, 0] = predictions

    inversed_data = scaler.inverse_transform(dummy_data)
    predictions_inversed = inversed_data[:, 0]  # İlk sütun tahminleri içeriyor

    print(predictions_inversed.shape)

    # Submission dosyası oluşturma
    submission = pd.DataFrame({"id": test_id, "bg+1:00": predictions_inversed})
    submission.to_csv("submission.csv", index=False)
    print("Submission file created.")
'''
