import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.layers import Dropout
from keras import regularizers
import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta

# Download Microsoft stock data from its IPO date till now
data = yf.download('MSFT', start='1986-03-13', end='2024-01-01')

# Add new features
data['Open-Close'] = data['Open'] - data['Close']
data['High-Low'] = data['High'] - data['Low']
data['Day'] = data.index.dayofweek
data['Month'] = data.index.month
data['Year'] = data.index.year
data['Volume Change'] = data['Volume'].diff()
data['Price Change'] = data['Close'].diff()
data['5-day MA'] = data['Close'].rolling(window=5).mean()

# Drop missing values
data = data.dropna()

# Keep only the 'Close' and the new features
data = data[['Close', 'Open-Close', 'High-Low', 'Day', 'Month', 'Year', 'Volume Change', 'Price Change', '5-day MA']]

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = pd.DataFrame(scaler.fit_transform(data), columns=data.columns, index=data.index)

# Create a separate scaler for 'Close' prices
close_scaler = MinMaxScaler(feature_range=(0, 1))
close_scaler.fit(data[['Close']])

# Split into training and testing sets (80% training, 20% testing)
train_size = int(len(data_scaled) * 0.8)
train, test = data_scaled.iloc[0:train_size, :], data_scaled.iloc[train_size:len(data_scaled), :]

# Function to create a dataset for LSTM
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset.iloc[i:(i+look_back), 1:]  # Use only the features for input
        X.append(a.values)
        Y.append(dataset.iloc[i + look_back, 0])  # Use 'Close' for output
    return np.array(X), np.array(Y)

# Create datasets for training and testing
look_back = 1
X_train, Y_train = create_dataset(train, look_back)
X_test, Y_test = create_dataset(test, look_back)

# Reshape input to be [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))

# Adjust the learning rate
optimizer = Adam(learning_rate=0.0001)

model = Sequential()
model.add(LSTM(250, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]), bias_regularizer=regularizers.l2(0.05)))
model.add(Dropout(0.2))
model.add(LSTM(250, bias_regularizer=regularizers.l2(0.05)))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer=optimizer)

# Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

# Train the model
model.fit(X_train, Y_train, epochs=100, batch_size=64, verbose=2, validation_split=0.2, callbacks=[early_stopping])

# Predict the stock prices on the testing data
predictions = model.predict(X_test)
predictions = close_scaler.inverse_transform(predictions)  # Use the 'close_scaler' here

# Compare the predicted prices with the actual values
Y_test = close_scaler.inverse_transform([Y_test])  # Use the 'close_scaler' here
plt.plot(Y_test[0])
plt.plot(predictions[:, 0])
plt.legend(['Actual', 'Predicted'])
plt.show()

# Print Predictions vs Actual
print('Predictions vs Actual')
print(pd.DataFrame({'Predicted': predictions[:, 0], 'Actual': Y_test[0]}))

# Test on unseen data
unseen_data = data_scaled.iloc[-look_back:, 1:]  # Use only the features for input
unseen_data = np.reshape(unseen_data.values, (1, unseen_data.shape[0], unseen_data.shape[1]))
unseen_prediction = model.predict(unseen_data)
unseen_prediction = close_scaler.inverse_transform(unseen_prediction)  # Use the 'close_scaler' here
print('Unseen Prediction:', unseen_prediction[0][0])

# Plot train loss vs validation loss
history = model.fit(X_train, Y_train, epochs=100, batch_size=64, verbose=2, validation_split=0.2, callbacks=[early_stopping])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Print overfitting or not
if history.history['val_loss'][-1] < history.history['loss'][-1]:
    print('The model is not overfitting')
else:
    print('The model is overfitting')