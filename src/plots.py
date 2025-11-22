import pandas as pd
import matplotlib.pyplot as plt

def plot_rolling_accuracy(y_test, y_pred, window=20):
    results = pd.DataFrame(index=y_test.index)
    results["Actual"] = y_test
    results["Predicted"] = y_pred
    results["Correct"] = (results["Actual"] == results["Predicted"]).astype(int)
    results["Rolling_Accuracy"] = results["Correct"].rolling(window).mean()

    plt.figure(figsize=(12, 5))
    results["Rolling_Accuracy"].plot(label=f"{window}-Day Rolling Accuracy", color="purple")
    plt.axhline(0.5, color="gray", linestyle="--", label="Random Guess (50%)")
    plt.title("NVDA Prediction Rolling Accuracy")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_price_with_predictions(data, y_test, y_pred):
    plt.figure(figsize=(14, 6))
    data.loc[y_test.index, "Close"].plot(label="NVDA Price", color="black")

    up_days = data.loc[y_test.index][y_pred == 1]
    down_days = data.loc[y_test.index][y_pred == 0]

    plt.scatter(up_days.index, up_days["Close"], color="green", label="Predicted UP", s=20)
    plt.scatter(down_days.index, down_days["Close"], color="red", label="Predicted DOWN", s=20)

    plt.title("NVDA Price with Model Predictions (UP vs DOWN)")
    plt.legend()
    plt.tight_layout()
    plt.show()
