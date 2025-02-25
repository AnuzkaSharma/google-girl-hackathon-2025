import os
import time
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ✅ Load Test Dataset
dataset_path = "D:/Anushka/Projects/Google Girl Hackathon/Training/CSV Files/feature_data.csv"
df = pd.read_csv(dataset_path)

# ✅ Select Features & Target
feature_cols = ['Total Fan-In', 'Total Fan-Out', 'Total Gate Count', 'Estimated Delay']
target_col = 'Estimated Depth'
X, y = df[feature_cols], df[target_col]

# ✅ Load Scaler & Apply Transformation
scaler_path = "D:/Anushka/Projects/Google Girl Hackathon/Training/saved_models/mlp_scaler.pkl"
scaler = joblib.load(scaler_path)
X_scaled = scaler.transform(X)

# Convert scaled data back to DataFrame with column names
X_test = pd.DataFrame(X_scaled, columns=feature_cols)

# ✅ Load Models
model_paths = {
    "Random Forest": "D:/Anushka/Projects/Google Girl Hackathon/Training/saved_models/best_random.pkl",
    "XGBoost": "D:/Anushka/Projects/Google Girl Hackathon/Training/saved_models/best_xgb.pkl",
    "MLP": "D:/Anushka/Projects/Google Girl Hackathon/Training/saved_models/best_mlp.pkl"
}

models = {name: joblib.load(path) for name, path in model_paths.items()}

# ✅ Evaluate Models
def evaluate_model(model, X, y):
    start_time = time.time()
    predictions = model.predict(X)
    end_time = time.time()
    inference_time = (end_time - start_time) / len(X)

    return {
        "MSE": mean_squared_error(y, predictions),
        "MAE": mean_absolute_error(y, predictions),
        "R² Score": r2_score(y, predictions),
        "Inference Time (ms)": inference_time * 1000
    }

comparison_results = {name: evaluate_model(model, X_test, y) for name, model in models.items()}

# ✅ Save & Display Results
results_df = pd.DataFrame(comparison_results).T
results_path = "D:/Anushka/Projects/Google Girl Hackathon/Training/saved_models/model_comparison.csv"
results_df.to_csv(results_path, index=True)

print("\n📊 Model Comparison Results:\n", results_df)
print(f"✅ Results saved at: {results_path}")
