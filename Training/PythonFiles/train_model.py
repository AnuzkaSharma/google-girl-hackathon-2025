import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

def load_dataset(csv_path):
    """
    Loads the extracted feature dataset from CSV and prepares the data for ML training.
    
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

# Main execution
if __name__ == "__main__":
    CSV_PATH = r"D:\Anushka\Projects\Google Girl Hackathon\Training\CSV Files\feature_data.csv"

    # Load dataset
    dataset = load_dataset(CSV_PATH)

    if dataset is not None:
        # Train separate models for depth and delay
        X = dataset[["Total Fan-In", "Total Fan-Out", "Total Gate Count"]]

        for target in ["Estimated Depth", "Estimated Delay"]:
            y = dataset[target]

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            # Evaluate model
            predictions = model.predict(X_test)
            mae = mean_absolute_error(y_test, predictions)

            print(f"\n✅ Model Training Complete for {target}!")
            print(f"📌 Mean Absolute Error: {mae:.4f}")
