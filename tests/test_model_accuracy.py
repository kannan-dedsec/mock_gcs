# tests/test_inference_parquet.py

import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

def test_model_accuracy_from_parquet():
    # Load processed dataset
    df = pd.read_parquet("processed/stocks.parquet")

    # Make sure expected columns exist
    required_cols = ["open", "high", "low", "close", "volume", "ma_10", "target"]
    for col in required_cols:
        assert col in df.columns, f"Missing column: {col}"

    # Prepare features and labels
    X = df[["open", "high", "low", "close", "volume", "ma_10"]]
    y_true = df["target"]

    # Load trained model
    model = joblib.load("model.joblib")

    # Run predictions
    y_pred = model.predict(X)

    # Compute accuracy
    acc = accuracy_score(y_true, y_pred)
    print(f"✅ Model accuracy on processed parquet: {acc:.4f}")

    # Save metrics for CML report
    with open("accuracy.txt", "w") as f:
        f.write(str(acc))

    # Test should fail if accuracy is below threshold
    assert acc > 0.6, f"Model accuracy too low: {acc:.2f}"
