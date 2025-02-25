import pandas as pd
import joblib  # For loading saved models

# ✅ Step 1: Load the trained models
depth_model = joblib.load("random_forest_estimated_depth.pkl")
delay_model = joblib.load("random_forest_estimated_delay.pkl")

# ✅ Step 2: Define Test Data (New Circuit Example)
test_data = pd.DataFrame({
    "Total Fan-In": [500],
    "Total Fan-Out": [500],
    "Total Gate Count": [200]
})

# ✅ Step 3: Predict Depth & Delay
predicted_depth = depth_model.predict(test_data)[0]
predicted_delay = delay_model.predict(test_data)[0]

# ✅ Step 4: Print the predictions
print("\n🎯 **Predictions for New Circuit:**")
print(f"📌 Estimated Depth: {predicted_depth:.2f}")
print(f"📌 Estimated Delay: {predicted_delay:.2f} ns")
