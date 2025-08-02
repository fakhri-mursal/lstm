# LSTM Stock Price Prediction

This repository contains Python scripts for training and testing an **LSTM (Long Short-Term Memory)** model to predict stock prices.  
The model uses historical stock data downloaded from Yahoo Finance, performs technical indicator calculations, and trains a deep learning model to forecast future prices.

---

## 📌 Features
- Downloads stock price data using `yfinance`
- Calculates technical indicators (RSI, SMA, EMA, etc.) using `pandas_ta`
- Normalizes data with `MinMaxScaler`
- Trains an LSTM model using TensorFlow/Keras
- Supports early stopping to prevent overfitting
- Saves predictions and visualization as HTML and image files

---

## 📦 Required Libraries

Before running the scripts, install the following Python libraries:

```bash
pip install tensorflow
pip install yfinance
pip install pandas
pip install numpy
pip install pandas-ta
pip install matplotlib
