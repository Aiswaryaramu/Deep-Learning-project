import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Attention, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam


def generate_time_series(n_points=12000):
    time = np.arange(n_points)
    trend = 0.0005 * time
    seasonality = 10 * np.sin(2 * np.pi * time / 50)
    noise = np.random.normal(0, 2, n_points)
    return 50 + trend + seasonality + noise


def create_sequences(series, lookback=30):
    X, y = [], []
    for i in range(len(series) - lookback):
        X.append(series[i:i + lookback])
        y.append(series[i + lookback])
    return np.array(X)[..., np.newaxis], np.array(y)


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


series = generate_time_series()
series = (series - series.mean()) / series.std()

X, y = create_sequences(series)
split = int(len(X) * 0.8)

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]


baseline_input = Input(shape=(X.shape[1], 1))
baseline_lstm = LSTM(64)(baseline_input)
baseline_output = Dense(1)(baseline_lstm)

lstm_model = Model(baseline_input, baseline_output)
lstm_model.compile(optimizer=Adam(0.001), loss="mse")
lstm_model.fit(X_train, y_train, epochs=5, batch_size=64, verbose=1)

lstm_preds = lstm_model.predict(X_test)
print("LSTM RMSE:", rmse(y_test, lstm_preds.squeeze()))


attn_input = Input(shape=(X.shape[1], 1))
encoder_output = LSTM(64, return_sequences=True)(attn_input)
attention_output = Attention()([encoder_output, encoder_output])
context_vector = GlobalAveragePooling1D()(attention_output)
attn_output = Dense(1)(context_vector)

attn_model = Model(attn_input, attn_output)
attn_model.compile(optimizer=Adam(0.001), loss="mse")
attn_model.fit(X_train, y_train, epochs=5, batch_size=64, verbose=1)

attn_preds = attn_model.predict(X_test)
print("Attention RMSE:", rmse(y_test, attn_preds.squeeze()))


attention_extractor = Model(attn_input, context_vector)
sample_attention = attention_extractor.predict(X_test[:1])
print("Sample Attention Representation (first 10 values):")
print(sample_attention[0][:10])