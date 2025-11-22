# nvda-stock-predictor
NVIDIA Stock Prediction Model

# NVDA Stock Direction Prediction (Machine Learning Model)
Predicting NVIDIA’s next-day price direction (UP/DOWN) using machine learning and technical analysis.

## Overview
This project builds a machine learning classifier that predicts whether NVDA’s closing price will move up or down the next trading day. It uses:

- Historical OHLCV data via `yfinance`
- Technical indicators (RSI, MACD, Bollinger Bands, Moving Averages)
- Lagged returns
- A Random Forest classifier
- A chronological train/test split
- Performance visualizations and optional trading backtesting

The goal is to demonstrate end-to-end financial modeling and ML workflow relevant to quantitative finance and fintech engineering.

## Features Used
Trend Indicators:
- MA5, MA10, MA20

Momentum:
- RSI (14-day)
- MACD
- MACD Signal
- MACD Histogram

Volatility:
- 10-day standard deviation of returns

Price Positioning:
- Bollinger Bands (Upper, Middle, Lower)
- Bollinger Percent

Price History:
- Daily return
- Lag1, Lag2, Lag3 returns

Volume:
- Raw daily trading volume

## Model
Random Forest Classifier:
- 200 trees
- Max depth = 6
- Trained chronologically (no data leakage)
- Compared against a baseline that always predicts “UP”

Model outputs:
- Accuracy
- Classification report
- Rolling accuracy plot
- Price chart with predictions overlaid

## Visualizations
Included in the project:
- Rolling 20-day accuracy
- Price chart with UP/DOWN predictions
- Optional confusion matrix
- Optional feature importance plot

These charts help evaluate model consistency and behavior over time.

## Backtesting (Optional)
A simple strategy simulation:
- Buy NVDA when the model predicts UP
- Stay in cash when it predicts DOWN

Outputs:
- Strategy returns
- Buy-and-hold comparison
- Sharpe ratio
- Max drawdown
- Win rate

This tests whether the model is not just accurate, but profitable.

## Installation
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
