import pandas as pd
import numpy as np
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from lightgbm import LGBMRegressor
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

def run_hyperion_training():
    """
    Trains the final Hyperion model using the enriched dataset.
    """
    print("--- Hyperion Model Training Pipeline Started ---")

    print("1/4 - Loading enriched dataset: hyperion_dataset.csv...")
    try:
        df = pd.read_csv("hyperion_dataset.csv")
    except FileNotFoundError:
        print("ERROR: 'hyperion_dataset.csv' not found. Please run build_features.py first.")
        return

    print("2/4 - Preprocessing data for modeling...")
    
    df['order_date_dt'] = pd.to_datetime(df['order_date_dt'])
    
    df['Delay'] = df['Days for shipping (real)'] - df['Days for shipment (scheduled)']
    df = df[df['Delay'] > 0]

    delay_cap = df['Delay'].quantile(0.99)
    df = df[df['Delay'] <= delay_cap]

    categorical_cols = [col for col in df.columns if df[col].dtype == 'object']
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    print("3/4 - Splitting data...")

    X = df.drop(columns=['Delay', 'Days for shipping (real)', 'Days for shipment (scheduled)', 'order_date_dt'])
    y = df['Delay']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("4/4 - Training the powerful Hyperion model...")
    hyperion_model = LGBMRegressor(
        objective='huber',
        random_state=42,
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31
    )
    
    hyperion_model.fit(X_train, y_train)

    y_pred = hyperion_model.predict(X_test)
    metrics = {
        'R-squared': r2_score(y_test, y_pred),
        'Mean Absolute Error (MAE)': mean_absolute_error(y_test, y_pred),
        'Root Mean Squared Error (RMSE)': np.sqrt(mean_squared_error(y_test, y_pred))
    }

    print("\n--- HYPERION MODEL EVALUATION RESULTS ---")
    print(f"  R-squared: {metrics['R-squared']:.4f}")
    print(f"  MAE: {metrics['Mean Absolute Error (MAE)']:.4f} days")
    print("---------------------------------------\n")
    
    joblib.dump(hyperion_model, 'hyperion_model.pkl')
    joblib.dump(list(X.columns), 'hyperion_model_columns.pkl')
    with open('hyperion_model_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print("Hyperion model artifacts (pkl, columns, metrics) saved successfully!")

if __name__ == "__main__":
    run_hyperion_training()