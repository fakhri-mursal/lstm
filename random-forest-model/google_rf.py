import yfinance as yf 
import pandas as pd 
import numpy as np
import pandas_ta as ta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Download Google's stock data from its IPO date till now
data = yf.download('GOOGL', start='2004-08-19', end='2024-01-01')

# Add new features
data['RSI'] = ta.rsi(data['Close'])
data['OBV'] = ta.obv(data['Close'], data['Volume'])
data['ATR'] = ta.atr(data['High'], data['Low'], data['Close'])
data['Yearly_MA'] = data['Close'].rolling(window=256).mean()
data['Garman Klass Vol'] = ((np.log(data['High']) - np.log(data['Low'])) ** 2) / 2 - (2 * np.log(2) - 1) * ((np.log(data['Adj Close']) - np.log(data['Open'])) ** 2)
data['Day'] = data.index.dayofweek
data['Month'] = data.index.month
data['Year'] = data.index.year

# Drop missing values
data = data.dropna()

# Keep only the 'Close' and the new features
data = data[['Close', 'RSI', 'OBV', 'ATR', 'Yearly_MA', 'Garman Klass Vol', 'Day', 'Month', 'Year']]

# Create correlation matrix with taret 'Close'
correlation_matrix = data.corr()['Close'].sort_values(ascending=False)
print(correlation_matrix)

# Build Random Forest model to precict 'Close'

# Normalize the data
scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_Y = MinMaxScaler(feature_range=(0, 1))

data_X = data.drop('Close', axis=1)
data_Y = data[['Close']]

data_X_scaled = pd.DataFrame(scaler_X.fit_transform(data_X), columns=data_X.columns, index=data_X.index)
data_Y_scaled = pd.DataFrame(scaler_Y.fit_transform(data_Y), columns=data_Y.columns, index=data_Y.index)

data_scaled = pd.concat([data_X_scaled, data_Y_scaled], axis=1)

# Split into training, validation and test sets
X = data_scaled.drop('Close', axis=1)
y = data_scaled['Close']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Validate the model (if you have a validation set)
val_predictions = model.predict(X_val)
val_mse = mean_squared_error(y_val, val_predictions)
val_r2 = r2_score(y_val, val_predictions)
print(f'\nValidation MSE:', val_mse)
print('Validation R²:', val_r2)

print(f"\n")

# Evaluate the model on the test set
test_predictions = model.predict(X_test)
test_mse = mean_squared_error(y_test, test_predictions)
test_r2 = r2_score(y_test, test_predictions)
print('Test MSE:', test_mse)
print('Test R²:', test_r2)

print(f"\n")

# Compare training and test set performance
train_predictions = model.predict(X_train)
train_mse = mean_squared_error(y_train, train_predictions)
train_r2 = r2_score(y_train, train_predictions)
print('Train MSE:', train_mse)
print('Train R²:', train_r2)

# Calculate the absolute errors
errors = abs(y_test - test_predictions)

# Create the scatter plot
plt.figure(figsize=(10, 6))
scatter = plt.scatter(y_test, test_predictions, c=errors, alpha=0.8, cmap='viridis')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Random Forest Model Predictions')
# Add a colorbar
plt.colorbar(scatter, label='Absolute Error')
plt.show()

# Residual plot
residuals = y_test - test_predictions
plt.figure(figsize=(10, 6))
plt.scatter(test_predictions, residuals, alpha=0.5)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.axhline(y=0, color='r', linestyle='--')
plt.show()

# Feature importances
feature_importances = model.feature_importances_
features = X_train.columns
indices = np.argsort(feature_importances)
plt.figure(figsize=(10, 6))
plt.title('Feature Importances')
plt.barh(range(len(indices)), feature_importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()

# Time series plot
plt.figure(figsize=(15, 6))
plt.plot(data.index[-len(y_test):], y_test, label='Actual Prices')
plt.plot(data.index[-len(y_test):], test_predictions, label='Predicted Prices', alpha=0.7)
plt.title('Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Prices')
plt.legend()
plt.show()