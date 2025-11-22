import yfinance as yf
from src.features import add_return_and_target, add_technical_indicators, get_feature_columns
from src.model import train_model, predict_tomorrow
from src.plots import plot_rolling_accuracy, plot_price_with_predictions

def main():
    # 1. Download data
    data = yf.download("NVDA", start="2015-01-01")

    # 2. Features + target
    data = add_return_and_target(data)
    data = add_technical_indicators(data)
    feature_cols = get_feature_columns()

    # 3. Train model
    results = train_model(data, feature_cols)
    model = results["model"]

    print("\nAccuracy:", round(results["accuracy"], 3))
    print("Baseline (always UP):", round(results["baseline_accuracy"], 3))
    print("\nClassification report:\n", results["classification_report"])

    # 4. Predict tomorrow
    latest_row = data.iloc[-1]
    tomorrow_pred = predict_tomorrow(model, latest_row, feature_cols)
    print("\nTomorrow's predicted direction:", "UP 📈" if tomorrow_pred == 1 else "DOWN 📉")

    # 5. Plots
    plot_rolling_accuracy(results["y_test"], results["y_pred"])
    plot_price_with_predictions(data, results["y_test"], results["y_pred"])

if __name__ == "__main__":
    main()
