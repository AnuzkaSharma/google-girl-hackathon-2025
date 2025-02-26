import os
import joblib
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error

# ✅ Load Dataset
data_path = "D:/Anushka/Projects/Google Girl Hackathon/Training/CSV Files/feature_data.csv"
df = pd.read_csv(data_path)

# ✅ Select Features & Target
feature_cols = ['Total Fan-In', 'Total Fan-Out', 'Total Gate Count', 'Estimated Delay']
target_col = 'Estimated Depth'  # Confirmed target column
X, y = df[feature_cols], df[target_col]

# ✅ Normalize Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ✅ Split Data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ✅ Initialize & Train MLP Model
mlp = MLPRegressor(hidden_layer_sizes=(128, 64, 32), activation='relu', solver='adam',
                    learning_rate_init=0.001, alpha=0.0001, max_iter=2000, random_state=42)
print("🚀 Training MLP Model...")
mlp.fit(X_train, y_train)

# ✅ Evaluate Model
train_mse = mean_squared_error(y_train, mlp.predict(X_train))
test_mse = mean_squared_error(y_test, mlp.predict(X_test))
cross_val_mse = cross_val_score(mlp, X_scaled, y, cv=5, scoring='neg_mean_squared_error')
cross_val_mse = -cross_val_mse.mean()

print(f"✅ Training MSE: {train_mse:.4f}")
print(f"✅ Testing MSE: {test_mse:.4f}")
print(f"📊 Cross-Validation MSE: {cross_val_mse:.4f}")

# ✅ Save Model & Scaler
model_path = "D:/Anushka/Projects/Google Girl Hackathon/Training/saved_models/best_mlp.pkl"
scaler_path = "D:/Anushka/Projects/Google Girl Hackathon/Training/saved_models/mlp_scaler.pkl"
os.makedirs(os.path.dirname(model_path), exist_ok=True)
joblib.dump(mlp, model_path)
joblib.dump(scaler, scaler_path)
print(f"💾 Model saved at: {model_path}")
print(f"💾 Scaler saved at: {scaler_path}")

print("🏆 MLP Training Completed!")
