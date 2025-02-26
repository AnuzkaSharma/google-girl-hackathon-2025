# 📌 Project Name: [Your Project Name]

## 🚀 Project Overview

Have you ever waited *forever* for a synthesis tool to give you results? 😩 Well, our project changes that! Instead of running a complete synthesis (which takes a lot of time), we use **machine learning** to quickly predict the **combinational depth** of crucial signals in an RTL module. 

### 🔹 Why does this matter?
- **Time-saving:** Instead of running full synthesis, we get quick predictions.
- **Feature-based learning:** We analyze key signal parameters like Fan-In, Fan-Out, and Gate Count.
- **Higher efficiency:** Our ML model gives accurate results without the long computation time.

---

## ⚙️ Installation & Setup

Even if you're new to this, setting up is *super easy*! 🚀 Just follow these steps:

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/AnuzkaSharma/google-girl-hackathon-2025.git
cd google-girl-hackathon-2025
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Download Pretrained Model & Scaler
```bash
wget https://raw.githubusercontent.com/AnuzkaSharma/google-girl-hackathon-2025/main/Training/saved_models/best_mlp.pkl -O best_mlp.pkl
wget https://raw.githubusercontent.com/AnuzkaSharma/google-girl-hackathon-2025/main/Training/saved_models/mlp_scaler.pkl -O mlp_scaler.pkl
```

---

## 🎯 How to Use?

### 1️⃣ Run the Model
```bash
python run_prediction.py --input sample_input.csv
```

### 2️⃣ Input Format
Our model expects a CSV file with the following features:
```csv
Total Fan-In, Total Fan-Out, Total Gate Count, Estimated Delay
```

### 3️⃣ What do you get?
A quick prediction of combinational depth! 🎯 Example output:
```
🔮 Predicted Combinational Depth: 1.92
```

---

## 📊 Model Details

### 🔹 Why did we choose MLP (Multi-Layer Perceptron)?
- It's great for numerical prediction tasks like this.
- It captures complex relationships between input features.
- It performs better than traditional regression models for our dataset.

### 🔹 Feature Engineering – What's important?
| Feature           | Importance |
|------------------|------------|
| Total Fan-Out    | 0.78       |
| Estimated Delay  | 0.70       |
| Total Gate Count | 0.39       |
| Total Fan-In     | 0.22       |

### 🔹 How well does our model perform?
| Metric  | Score |
|---------|------|
| MSE     | 0.0662 |
| MAE     | 0.0443 |
| R² Score| 0.98  |

---

## 🟢 Google Colab Demo (No setup needed!)
Click below to run the model **directly in Google Colab** (No installation required!) 🚀
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AnuzkaSharma/google-girl-hackathon-2025/blob/main/colab_demo.ipynb)

---

## 🤝 Want to Contribute?
If you have ideas or want to improve the model, **feel free to create a pull request!** 🚀

🔹 **GitHub:** [Your GitHub Profile](https://github.com/AnuzkaSharma)  
🔹 **LinkedIn:** [Your LinkedIn Profile](https://linkedin.com/in/yourprofile)

