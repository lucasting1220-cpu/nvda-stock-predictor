NVIDIA Stock Prediction Model

Predicting NVIDIA’s next-day price direction (UP/DOWN) using machine learning and technical analysis.

## Overview
This project builds a machine learning classifier that predicts whether NVDA’s closing price will move up or down the next trading day. It uses:

- Technical indicators (RSI, MACD, Bollinger Bands, Moving Averages)
- Lagged returns
- A Random Forest classifier
- A chronological train/test split
- Performance visualizations and optional trading backtesting

The goal is to demonstrate end-to-end financial modeling and ML workflow relevant to quantitative finance and fintech engineering.

The features I used:
Trend Indicators:
- MA5, MA10, MA20

Price History:
- Daily return
- Lag1, Lag2, Lag3 returns

What was used for visualization
Included in the project:
- Rolling 20-day accuracy
- Price chart with UP/DOWN predictions
- Optional confusion matrix
- Optional feature importance plot


## Installation
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
