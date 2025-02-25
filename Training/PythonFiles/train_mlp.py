import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

# ----------------------------------------------
# ✅ Step 1: Load Dataset Function
# ----------------------------------------------
def get_dataset_path():
    """Returns the correct dataset path."""
    dataset_path = "D:/Anushka/Projects/Google Girl Hackathon/Training/CSV Files/feature_data.csv"

    if not os.path.exists(dataset_path):
        print(f"❌ Error: Dataset not found at {dataset_path}!")
        return None

    print(f"📂 Dataset found at: {dataset_path}")
    return dataset_path

def load_data():
    """Loads the dataset from the fixed path."""
    dataset_path = get_dataset_path()
    if dataset_path is None:
        return None

    try:
        df = pd.read_csv(dataset_path)
        print(f"✅ Successfully loaded dataset from {dataset_path}")
        return df
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None

# ----------------------------------------------
# ✅ Step 2: Data Preprocessing
# ----------------------------------------------
def preprocess_data(df, target_column):
    """Preprocess dataset: Extract features/target and normalize input."""
    if df is None:
        print("❌ Error: No dataset found!")
        return None, None, None

    print("🔄 Preprocessing dataset...")

    # ✅ Remove non-numeric columns if any
    non_numeric_cols = ["Filename", "Folder"]
    df = df.drop(columns=[col for col in non_numeric_cols if col in df.columns], errors="ignore")

    # ✅ Check if target column exists
    if target_column not in df.columns:
        print(f"❌ Error: Target column '{target_column}' not found in dataset!")
        return None, None, None

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # ✅ Check for extreme values before scaling
    print(f"🔍 Checking dataset values: Max={X.max().max()}, Min={X.min().min()}")

    # ✅ Normalize Features for Neural Network
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ✅ After Scaling, Check Values
    print(f"🔍 After Scaling: Max={X_scaled.max()}, Min={X_scaled.min()}")

    return X_scaled, y, scaler

# ----------------------------------------------
# ✅ Step 3: Train MLP Neural Network with Hyperparameter Tuning
# ----------------------------------------------
def train_mlp(X, y, scaler, model_save_path="D:/Anushka/Projects/Google Girl Hackathon/Training/saved_models/best_mlp.pkl"):
    """Trains an MLP Neural Network with hyperparameter tuning and saves it."""
    print("🚀 Initializing MLP Neural Network model with tuning...")

    # ✅ Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ✅ Define Base MLP Model
    mlp_model = MLPRegressor(max_iter=1000, random_state=42)

    # ✅ Define Hyperparameter Grid for RandomizedSearchCV
    param_dist = {
        "hidden_layer_sizes": [(64, 32), (128, 64, 32), (256, 128, 64)],
        "activation": ["relu", "tanh"],
        "solver": ["adam"],  # 👈 "sgd" hata diya kyunki instability la raha tha
        "learning_rate_init": [0.001, 0.01],
        "alpha": [0.0001, 0.001, 0.01]
    }

    # ✅ Use RandomizedSearchCV for Tuning
    random_search = RandomizedSearchCV(
        mlp_model, param_distributions=param_dist, 
        n_iter=10, cv=3, scoring="neg_mean_squared_error", 
        random_state=42, n_jobs=-1
    )

    print("🎯 Running Hyperparameter Tuning...")
    random_search.fit(X_train, y_train)

    # ✅ Best Model from Tuning
    best_mlp = random_search.best_estimator_
    best_params = random_search.best_params_
    print(f"🏆 Best Parameters Found: {best_params}")

    # ✅ Model Evaluation
    print("📊 Evaluating Best MLP Model...")
    y_train_pred = best_mlp.predict(X_train)
    y_test_pred = best_mlp.predict(X_test)

    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)

    print(f"✅ Training MSE: {train_mse:.4f}")
    print(f"✅ Testing MSE: {test_mse:.4f}")

    # ✅ Ensure folder exists before saving model
    model_dir = os.path.dirname(model_save_path)
    os.makedirs(model_dir, exist_ok=True)

    # ✅ Save Best Model & Scaler
    try:
        joblib.dump({"model": best_mlp, "scaler": scaler}, model_save_path)
        print(f"💾 Best tuned MLP model saved at: {model_save_path}")
    except Exception as e:
        print(f"❌ ERROR while saving model: {e}")

    return best_mlp, {"train_mse": train_mse, "test_mse": test_mse}

# ----------------------------------------------
# ✅ Step 4: Run the Training Process
# ----------------------------------------------
if __name__ == "__main__":
    print("📥 Loading dataset for MLP Neural Network training...")
    df = load_data()

    if df is not None:
        target_column = "Estimated Depth"  # 👈 Make sure this column exists in CSV
        X, y, scaler = preprocess_data(df, target_column)

        if X is not None and y is not None:
            best_mlp, results = train_mlp(X, y, scaler)
            print("\n🏆 Model Training & Tuning Completed!", results)
        else:
            print("❌ Training aborted due to preprocessing failure.")
    else:
        print("❌ Training aborted due to missing dataset.")
