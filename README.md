# SynthPredict: ML-Powered RTL Synthesis Analysis

## 📊 Project Overview

SynthPredict is an innovative machine learning solution that revolutionizes RTL synthesis workflows by accurately predicting combinational depth without executing complete synthesis runs.

Digital design engineers often face significant wait times during synthesis operations, creating bottlenecks in development pipelines. SynthPredict addresses this challenge by leveraging carefully crafted machine learning models to deliver near-instantaneous predictions of critical path metrics.

## 🔬 Technical Innovation

Our approach employs a Multi-Layer Perceptron (MLP) architecture, trained on a comprehensive dataset of synthesis results from diverse RTL modules. Through extensive feature engineering and model optimization, we've achieved remarkably accurate predictions with minimal computational overhead.

### Key Capabilities

- **Rapid Assessment**: Generate combinational depth predictions in milliseconds rather than minutes or hours
- **Feature-Driven Analysis**: Leverages key signal characteristics including Fan-In, Fan-Out, and Gate Count
- **High Accuracy**: Achieves 98% correlation with actual synthesis results
- **Design Workflow Integration**: Seamlessly fits into existing RTL development processes

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.8+
- pip package manager

### Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/AnuzkaSharma/google-girl-hackathon-2025.git
   cd google-girl-hackathon-2025
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download pre-trained model and scaler:
   ```bash
   wget https://raw.githubusercontent.com/AnuzkaSharma/google-girl-hackathon-2025/main/Training/saved_models/best_mlp.pkl -O best_mlp.pkl
   wget https://raw.githubusercontent.com/AnuzkaSharma/google-girl-hackathon-2025/main/Training/saved_models/mlp_scaler.pkl -O mlp_scaler.pkl
   ```

## 🚀 Usage Guide

### Running Predictions

Execute the prediction tool with your input data:
```bash
python run_prediction.py --input sample_input.csv
```

### Input Format Requirements

The model requires a CSV file containing the following features:
```
Total Fan-In, Total Fan-Out, Total Gate Count, Estimated Delay
```

### Output Example

```
Analysis complete:
Predicted Combinational Depth: 1.92 (±0.07)
Confidence Score: 98.4%
```

## 📈 Model Performance

### Architecture Selection
After extensive experimentation with various models including Linear Regression, Random Forest, and Gradient Boosting, we determined that a Multi-Layer Perceptron provides the optimal balance of accuracy and inference speed for this specific problem domain.

### Feature Importance

| Feature | Relative Importance |
|---------|---------------------|
| Total Fan-Out | 0.78 |
| Estimated Delay | 0.70 |
| Total Gate Count | 0.39 |
| Total Fan-In | 0.22 |

### Performance Metrics

| Metric | Score |
|--------|-------|
| Mean Squared Error (MSE) | 0.0662 |
| Mean Absolute Error (MAE) | 0.0443 |
| R² Score | 0.98 |

## 🔍 Interactive Demo

Experience SynthPredict immediately via our interactive Google Colab notebook:
[Launch SynthPredict Demo]((https://colab.research.google.com/github/AnuzkaSharma/google-girl-hackathon-2025/blob/main/GGH_colab_demo.ipynb))

## 🤝 Contribution Guidelines

We welcome contributions to enhance model accuracy, feature set, or documentation. Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Submit a pull request with detailed description

## 📧 Contact Information

- **GitHub:** ((https://github.com/AnuzkaSharma))
- **LinkedIn:** ((https://www.linkedin.com/in/anuzka/))

## 📚 References

Dataset taken from:
Open-Source RTL Benchmark Datasets (Recommended)
✅ ISCAS85, ISCAS89, ITC99 Benchmarks
📌 ISCAS & ITC Benchmark Circuits
➡ Verilog/VHDL circuits ke benchmarks jo testing ke liye use ho sakte hain!


Tools & Documentation
Scikit-learn: https://scikit-learn.org/stable/ – Used for training & evaluating ML models.
SHAP for Explainability: https://shap.readthedocs.io/en/latest/ – Helps in understanding model predictions.
XGBoost & Feature Importance: https://xgboost.readthedocs.io/en/stable/ – One of the benchmark models we compared.
