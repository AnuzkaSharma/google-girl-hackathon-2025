import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

# 📌 Correct model & data paths
model_path = "D:/Anushka/Projects/Google Girl Hackathon/Training/saved_models/mlp_model.pkl"
data_path = "D:/Anushka/Projects/Google Girl Hackathon/Training/saved_models/mlp_feature_importance.csv"

# ✅ Check if model file exists before loading
import os
if not os.path.exists(model_path):
    raise FileNotFoundError(f"🚨 Model file not found at: {model_path}\n🔄 Train & save the model first!")

# 🔍 Load trained model
mlp_model = joblib.load(model_path)

# 🔍 Load feature importance data
df_importance = pd.read_csv(data_path)

# 📊 Visualize Feature Importance
print("📊 Feature Importance Data:")
print(df_importance)

plt.figure(figsize=(8, 4))
plt.barh(df_importance["Feature"], df_importance["Importance"], color="skyblue")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.title("🔍 Feature Importance Visualization")
plt.gca().invert_yaxis()
plt.show()
