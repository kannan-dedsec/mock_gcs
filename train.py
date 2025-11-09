
from feast import FeatureStore
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

store = FeatureStore(repo_path="feature_repo")

offline_df = pd.read_parquet("processed/stocks.parquet")

X = offline_df[["open", "high", "low", "close", "volume", "ma_10"]]
y = offline_df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)
print("Model accuracy:", model.score(X_test, y_test))

# Save model
joblib.dump(model, "model.joblib")
print("✅ Model saved to model.joblib")
