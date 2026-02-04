import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Attention
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA


np.random.seed(42)
time = np.arange(0, 500)
trend = 0.05 * time
seasonality = 10 * np.sin(2 * np.pi * time / 50)
noise = np.random.normal(0, 2, size=len(time))

series = trend + seasonality + noise
data = pd.DataFrame({"value": series})


def create_sequences(data, lookback=30):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i+lookback])
        y.append(data[i+lookback])
    return np.array(X), np.array(y)

LOOKBACK = 30
X, y = create_sequences(series, LOOKBACK)

X = X.reshape((X.shape[0], X.shape[1], 1))
y = y.reshape((-1, 1))

def rolling_origin_split(X, y, initial_train=300, horizon=50):
    splits = []
    for start in range(initial_train, len(X)-horizon, horizon):
        X_train, y_train = X[:start], y[:start]
        X_test, y_test = X[start:start+horizon], y[start:start+horizon]
        splits.append((X_train, y_train, X_test, y_test))
    return splits

splits = rolling_origin_split(X, y)


def build_attention_model(input_shape):
    encoder_inputs = Input(shape=input_shape)
    encoder_lstm = LSTM(64, return_sequences=True, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)

    decoder_inputs = Input(shape=input_shape)
    decoder_lstm = LSTM(64, return_sequences=True)
    decoder_outputs = decoder_lstm(decoder_inputs, initial_state=[state_h, state_c])

    attention = Attention()
    context = attention([decoder_outputs, encoder_outputs])

    concat = tf.keras.layers.Concatenate()([decoder_outputs, context])
    output = Dense(1)(concat[:, -1, :])

    model = Model([encoder_inputs, decoder_inputs], output)
    model.compile(optimizer="adam", loss="mse")

    return model


results = []

for fold, (X_train, y_train, X_test, y_test) in enumerate(splits):
    model = build_attention_model((LOOKBACK, 1))

    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    model.fit(
        [X_train, X_train],
        y_train,
        validation_split=0.2,
        epochs=40,
        batch_size=32,
        callbacks=[early_stop],
        verbose=0
    )

    preds = model.predict([X_test, X_test])
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)

    results.append({"Fold": fold+1, "RMSE": rmse, "MAE": mae})

results_df = pd.DataFrame(results)
print("\nDeep Learning Results:\n", results_df)


train_arima = series[:350]
test_arima = series[350:400]

arima_model = ARIMA(train_arima, order=(5,1,0)).fit()
arima_preds = arima_model.forecast(steps=len(test_arima))

arima_rmse = np.sqrt(mean_squared_error(test_arima, arima_preds))
arima_mae = mean_absolute_error(test_arima, arima_preds)

print("\nARIMA RMSE:", arima_rmse)
print("ARIMA MAE:", arima_mae)


plt.figure(figsize=(10,5))
plt.plot(series, label="Actual")
plt.plot(range(350, 400), arima_preds, label="ARIMA Forecast")
plt.legend()
plt.title("Time Series Forecasting Comparison")
plt.show()