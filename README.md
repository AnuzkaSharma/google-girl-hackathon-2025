📌 Project Name: [Your Project Name]

🚀 Project Overview

This project aims to predict the combinational depth of crucial pre-identified signals in an RTL module using machine learning. Instead of running a full synthesis (which is time-consuming), our model provides quick estimations using extracted features.

🔹 Objective

Reduce synthesis run-time by providing ML-based combinational depth prediction.

Use feature engineering to extract impactful parameters (Fan-In, Fan-Out, etc.).

Ensure high accuracy & fast inference for real-world usability.

⚙️ Installation & Setup

1️⃣ Clone the Repository

git clone https://github.com/AnuzkaSharma/google-girl-hackathon-2025.git
cd google-girl-hackathon-2025

2️⃣ Install Dependencies

pip install -r requirements.txt

3️⃣ Download Pretrained Model & Scaler

wget https://raw.githubusercontent.com/AnuzkaSharma/google-girl-hackathon-2025/main/Training/saved_models/best_mlp.pkl -O best_mlp.pkl
wget https://raw.githubusercontent.com/AnuzkaSharma/google-girl-hackathon-2025/main/Training/saved_models/mlp_scaler.pkl -O mlp_scaler.pkl

🎯 Usage Guide

1️⃣ Run the Model

python run_prediction.py --input sample_input.csv

2️⃣ Input Format

The model requires a CSV file with the following features:

Total Fan-In, Total Fan-Out, Total Gate Count, Estimated Delay

3️⃣ Output

Predicted Combinational Depth

Example Output:

🔮 Predicted Combinational Depth: 1.92

📊 Model Details

🔹 ML Approach

Model Used: MLP Regressor

Hyperparameter Tuning: GridSearchCV

🔹 Feature Engineering

Feature

Importance

Total Fan-Out

0.78

Estimated Delay

0.70

Total Gate Count

0.39

Total Fan-In

0.22

🔹 Evaluation Metrics

Metric

Score

MSE

0.0662

MAE

0.0443

R² Score

0.98

🟢 Google Colab Demo

Run the model directly on Google Colab (No local setup required!)


🤝 Contributing & Contact

If you have any suggestions or want to contribute, feel free to create a pull request! 🚀

🔹 GitHub: Your GitHub Profile🔹 LinkedIn: Your LinkedIn Profile

