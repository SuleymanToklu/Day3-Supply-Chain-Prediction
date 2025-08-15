import pandas as pd
import numpy as np
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

def save_artifacts(model, columns, metrics, prefix):
    joblib.dump(model, f'{prefix}_model.pkl')
    joblib.dump(columns, f'{prefix}_model_columns.pkl')
    with open(f'{prefix}_model_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"--- Artifacts for '{prefix}' model saved successfully! ---")

def run_dual_training_pipeline():
    print("--- Dual Model Training Pipeline Started ---")

    print("\n1/3 - Loading Data...")
    try:
        df = pd.read_csv("DataCoSupplyChainDataset.csv", encoding='latin1')
    except FileNotFoundError:
        print("ERROR: 'DataCoSupplyChainDataset.csv' not found.")
        return
    
    # --------------------------------------------------------------------------
    # 2. TRAIN BASE MODEL (Simple XGBoost)
    # --------------------------------------------------------------------------
    print("\n2/3 - Training BASELINE Model (XGBoost)...")
    
    base_features = [
        'Type', 'Delivery Status', 'Category Name', 'Customer Country',
        'Market', 'Order Country', 'Order Region', 'Shipping Mode',
        'Days for shipping (real)', 'Days for shipment (scheduled)'
    ]
    df_base = df[base_features].copy()
    df_base.dropna(inplace=True) # FIX: Drop NA only from the selected columns for this model
    
    df_base['Delay'] = df_base['Days for shipping (real)'] - df_base['Days for shipment (scheduled)']
    df_base = df_base[df_base['Delay'] > 0]

    base_categorical_cols = [col for col in df_base.columns if df_base[col].dtype == 'object']
    for col in base_categorical_cols:
        le = LabelEncoder()
        df_base[col] = le.fit_transform(df_base[col])

    X_base = df_base.drop(['Delay', 'Days for shipping (real)', 'Days for shipment (scheduled)'], axis=1)
    y_base = df_base['Delay']
    X_train_base, X_test_base, y_train_base, y_test_base = train_test_split(X_base, y_base, test_size=0.2, random_state=42)

    base_model = XGBRegressor(random_state=42, n_estimators=100, max_depth=5)
    base_model.fit(X_train_base, y_train_base)
    
    y_pred_base = base_model.predict(X_test_base)
    base_metrics = {
        'R-squared': r2_score(y_test_base, y_pred_base),
        'Mean Absolute Error (MAE)': mean_absolute_error(y_test_base, y_pred_base),
        'Root Mean Squared Error (RMSE)': np.sqrt(mean_squared_error(y_test_base, y_pred_base))
    }
    
    print("Baseline Model Performance:", base_metrics)
    save_artifacts(base_model, list(X_base.columns), base_metrics, 'base')

    # --------------------------------------------------------------------------
    # 3. TRAIN ADVANCED MODEL (LightGBM with Feature Engineering)
    # --------------------------------------------------------------------------
    print("\n3/3 - Training ADVANCED Model (LightGBM)...")

    adv_features = [
        'Type', 'Delivery Status', 'Category Name', 'Customer Country', 'Market', 
        'Order Country', 'Order Region', 'Shipping Mode', 'Sales', 'Order Item Quantity',
        'order date (DateOrders)', 'Days for shipping (real)', 'Days for shipment (scheduled)'
    ]
    df_adv = df[adv_features].copy()
    df_adv.dropna(inplace=True) # FIX: Drop NA only from the selected columns for this model

    df_adv['Delay'] = df_adv['Days for shipping (real)'] - df_adv['Days for shipment (scheduled)']
    df_adv = df_adv[df_adv['Delay'] > 0]
    
    delay_cap = df_adv['Delay'].quantile(0.99)
    df_adv = df_adv[df_adv['Delay'] <= delay_cap]

    df_adv['order date (DateOrders)'] = pd.to_datetime(df_adv['order date (DateOrders)'], errors='coerce')
    df_adv.dropna(subset=['order date (DateOrders)'], inplace=True)
    
    df_adv['order_month'] = df_adv['order date (DateOrders)'].dt.month
    df_adv['order_day_of_week'] = df_adv['order date (DateOrders)'].dt.dayofweek
    df_adv['market_shipping_interaction'] = df_adv['Market'].astype(str) + '_' + df_adv['Shipping Mode'].astype(str)
    
    adv_categorical_cols = [col for col in df_adv.columns if df_adv[col].dtype == 'object' and col != 'order date (DateOrders)']
    adv_categorical_cols.append('market_shipping_interaction') # Also encode the new interaction feature
    
    for col in adv_categorical_cols:
        le = LabelEncoder()
        df_adv[col] = le.fit_transform(df_adv[col])
    
    X_adv = df_adv.drop(['Delay', 'Days for shipping (real)', 'Days for shipment (scheduled)', 'order date (DateOrders)'], axis=1)
    y_adv = df_adv['Delay']
    X_train_adv, X_test_adv, y_train_adv, y_test_adv = train_test_split(X_adv, y_adv, test_size=0.2, random_state=42)

    advanced_model = LGBMRegressor(objective='huber', random_state=42, n_estimators=300)
    advanced_model.fit(X_train_adv, y_train_adv)

    y_pred_adv = advanced_model.predict(X_test_adv)
    advanced_metrics = {
        'R-squared': r2_score(y_test_adv, y_pred_adv),
        'Mean Absolute Error (MAE)': mean_absolute_error(y_test_adv, y_pred_adv),
        'Root Mean Squared Error (RMSE)': np.sqrt(mean_squared_error(y_test_adv, y_pred_adv))
    }
    
    print("Advanced Model Performance:", advanced_metrics)
    save_artifacts(advanced_model, list(X_adv.columns), advanced_metrics, 'advanced')

    print("\n--- Dual Training Pipeline Completed Successfully! ---")

if __name__ == "__main__":
    run_dual_training_pipeline()