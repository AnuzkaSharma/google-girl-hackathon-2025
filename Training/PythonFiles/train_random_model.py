import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV

def load_dataset(csv_path):
    """
    Loads the extracted feature dataset from CSV for ML training.
    
    Args:
    csv_path (str): Path to the CSV file
    
    Returns:
    DataFrame: Loaded dataset
    """
    try:
        df = pd.read_csv(csv_path)
        print("\n📌 Feature Data Loaded Successfully!")
        print(df.head())  # Display first 5 rows for verification
        return df
    except Exception as e:
        print(f"❌ Error loading dataset: {str(e)}")
        return None

def train_random_forest(X, y):
    """
    Trains a Random Forest model with hyperparameter tuning.
    
    Args:
    X (DataFrame): Feature matrix
    y (Series): Target variable
    
    Returns:
    model: Trained Random Forest model
    """
    # Split data into training (80%) and testing (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define parameter grid for tuning
    param_grid = {
        'n_estimators': [50, 100, 200],  # Number of trees
        'max_depth': [5, 10, 20, None],  # Maximum depth of trees
        'min_samples_split': [2, 5, 10]  # Minimum samples needed for a split
    }

    # GridSearchCV to find best parameters
    grid_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Best hyperparameters
    best_params = grid_search.best_params_
    print("✅ Best Random Forest Parameters:", best_params)

    # Train final model with best parameters
    rf_model = RandomForestRegressor(**best_params)
    rf_model.fit(X_train, y_train)

    # Evaluate model performance
    predictions = rf_model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)

    print(f"📌 Optimized Random Forest MAE: {mae:.4f}")

    return rf_model

# ✅ Main Execution
if __name__ == "__main__":
    CSV_PATH = r"D:\Anushka\Projects\Google Girl Hackathon\Training\CSV Files\feature_data.csv"

    # Load dataset
    dataset = load_dataset(CSV_PATH)

    if dataset is not None:
        X = dataset[["Total Fan-In", "Total Fan-Out", "Total Gate Count"]]

        for target in ["Estimated Depth", "Estimated Delay"]:
            print(f"\n🚀 Training Model for {target}...")
            y = dataset[target]

            # Train model
            model = train_random_forest(X, y)

            # Save Model (if required for future use)
            import joblib
            model_filename = f"random_forest_{target.lower().replace(' ', '_')}.pkl"
            joblib.dump(model, model_filename)
            print(f"✅ Model saved as: {model_filename}")
