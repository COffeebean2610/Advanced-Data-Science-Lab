# ==============================================================================
# Crop Recommendation System - Model Training
# B.Tech 7th Sem - Advanced Data Science Laboratory
# ==============================================================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

def main():
    print("--- Step 1: Loading Dataset ---")
    dataset_path = "Crop_recommendation.csv"
    df = pd.read_csv(dataset_path)
    print(f"Dataset loaded successfully with shape: {df.shape}")
    print(f"Features: {list(df.columns[:-1])}")
    print(f"Target: {df.columns[-1]}")
    print(f"Total unique crop types: {df['label'].nunique()}")

    print("\n--- Step 2: Splitting Data into Train and Test Sets ---")
    # Separating features (X) and target (y)
    X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y = df['label']

    # 80% training data and 20% testing data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

    print("\n--- Step 3: Training Random Forest Classifier ---")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("Model training completed.")

    print("\n--- Step 4: Model Evaluation ---")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy Score: {acc * 100:.2f}%")
    print("\nClassification Report Preview:")
    print(classification_report(y_test, y_pred, digits=4))

    print("--- Step 5: Saving Model ---")
    model_filename = "model.pkl"
    joblib.dump(model, model_filename)
    print(f"Trained model saved successfully as '{model_filename}'")

if __name__ == "__main__":
    main()
