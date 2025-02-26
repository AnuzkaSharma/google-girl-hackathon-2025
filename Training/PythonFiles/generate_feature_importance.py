import os
import joblib
import shap
import matplotlib.pyplot as plt
import pandas as pd

import os

# ✅ Ensure 'plots/' directory exists before saving
plot_dir = "D:/Anushka/Projects/Google Girl Hackathon/Training/plots"
os.makedirs(plot_dir, exist_ok=True)  

# ✅ Now save the plot correctly
plt.savefig(f"{plot_dir}/shap_summary_plot.png")


# ✅ Load Model & Data
model_path = "D:/Anushka/Projects/Google Girl Hackathon/Training/saved_models/best_mlp.pkl"
scaler_path = "D:/Anushka/Projects/Google Girl Hackathon/Training/saved_models/mlp_scaler.pkl"
data_path = "D:/Anushka/Projects/Google Girl Hackathon/Training/CSV Files/feature_data.csv"

df = pd.read_csv(data_path)
feature_cols = ['Total Fan-In', 'Total Fan-Out', 'Total Gate Count', 'Estimated Delay']
X = df[feature_cols]

# ✅ Scale Data
scaler = joblib.load(scaler_path)
X_scaled = scaler.transform(X)

# ✅ Load Model
mlp_model = joblib.load(model_path)


import matplotlib.pyplot as plt

# Improve color, add grid, and adjust text alignment
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis="x", linestyle="--", alpha=0.5)

# Adjust text color for better contrast
for text in plt.gca().texts:
    text.set_color("black")  # Change text color to black for better readability
    text.set_fontsize(12)     # Increase font size if needed

plt.tight_layout()
plt.savefig(f"{plot_dir}/shap_summary_plot.png", dpi=300, bbox_inches="tight")  # High quality save
plt.show()

# ✅ Compute SHAP Values
explainer = shap.KernelExplainer(mlp_model.predict, X_scaled[:50])  # Use small subset for efficiency
shap_values = explainer(X_scaled)

# ✅ Check SHAP Values Shape
print(f"SHAP Values Shape: {shap_values.shape}")




# ✅ First Plot: Summary Plot
plt.figure()
shap.summary_plot(shap_values, X_scaled, feature_names=feature_cols)
plt.savefig("D:/Anushka/Projects/Google Girl Hackathon/Training/plots/shap_summary_plot.png")
plt.close()

# ✅ Second Plot: Feature Importance
plt.figure()
shap.plots.bar(shap_values)
plt.savefig("D:/Anushka/Projects/Google Girl Hackathon/Training/plots/shap_feature_importance.png")
plt.close()

print("✅ SHAP Feature Importance Generated & Saved!")
