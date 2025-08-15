# app.py (Comparison Version)

import streamlit as st
import pandas as pd
import joblib
import json
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go
import numpy as np

# --- Page Settings ---
st.set_page_config(page_title="Model Karşılaştırma", page_icon="🔬", layout="wide")

# --- Resource Loading ---
@st.cache_resource
def load_all_resources():
    """Loads all artifacts for both base and advanced models."""
    resources = {}
    try:
        # Load Base Model
        resources['base_model'] = joblib.load('base_model.pkl')
        resources['base_model_columns'] = joblib.load('base_model_columns.pkl')
        with open('base_model_metrics.json', 'r') as f:
            resources['base_model_metrics'] = json.load(f)
        
        # Load Advanced Model
        resources['advanced_model'] = joblib.load('advanced_model.pkl')
        resources['advanced_model_columns'] = joblib.load('advanced_model_columns.pkl')
        with open('advanced_model_metrics.json', 'r') as f:
            resources['advanced_model_metrics'] = json.load(f)
            
    except FileNotFoundError:
        return None
    return resources

@st.cache_data
def load_raw_data():
    """Loads the raw data for dropdowns."""
    try:
        return pd.read_csv("DataCoSupplyChainDataset.csv", encoding='latin1')
    except FileNotFoundError:
        return None

resources = load_all_resources()
df_raw = load_raw_data()

# --- Main App ---
st.title("🔬 Tedarik Zinciri Gecikme Modeli Karşılaştırması")

if not resources or df_raw is None:
    st.error("Model dosyaları bulunamadı! Lütfen önce `train_model.py` betiğini çalıştırdığınızdan emin olun.")
    st.stop()

# --- Tab Layout ---
tab1, tab2 = st.tabs(["📊 **Karşılaştırmalı Tahmin Aracı**", "📈 **Model Performans Analizi**"])

with tab1:
    st.header("Canlı Gecikme Tahmini: Temel ve Gelişmiş Model")
    
    with st.form(key='prediction_form'):
        col1, col2, col3 = st.columns(3)
        with col1:
            market = st.selectbox('Pazar (Market)', options=sorted(df_raw['Market'].unique()))
            order_region = st.selectbox('Sipariş Bölgesi (Order Region)', options=sorted(df_raw['Order Region'].unique()))
        with col2:
            shipping_mode = st.selectbox('Kargo Türü (Shipping Mode)', options=sorted(df_raw['Shipping Mode'].unique()))
            category_name = st.selectbox('Ürün Kategorisi', options=sorted(df_raw['Category Name'].unique()))
        with col3:
            order_country = st.selectbox('Sipariş Ülkesi', options=sorted(df_raw['Order Country'].unique()))
            # We need Sales and Quantity for the advanced model
            sales = st.number_input('Satış Tutarı (Sales)', min_value=0.0, value=150.0)
            quantity = st.number_input('Ürün Adedi (Quantity)', min_value=1, value=3)

        submit_button = st.form_submit_button(label='Gecikmeleri Karşılaştır')

    if submit_button:
        # --- Prepare Input for Base Model ---
        base_input_df = pd.DataFrame([{
            'Market': market, 'Order Region': order_region, 'Shipping Mode': shipping_mode,
            'Category Name': category_name, 'Order Country': order_country
        }])
        # Include all other columns required by the base model (even if not in UI)
        for col in resources['base_model_columns']:
            if col not in base_input_df.columns:
                base_input_df[col] = 0 # Default value
        
        # --- Prepare Input for Advanced Model ---
        adv_input_df = base_input_df.copy() # Start with the same base
        adv_input_df['Sales'] = sales
        adv_input_df['Order Item Quantity'] = quantity
        adv_input_df['order_month'] = pd.to_datetime('today').month
        adv_input_df['order_day_of_week'] = pd.to_datetime('today').dayofweek
        adv_input_df['market_shipping_interaction'] = f"{market}_{shipping_mode}"
        
        # --- Encoding and Prediction ---
        try:
            # Encode Base
            base_predict_df = base_input_df.copy()
            for col in ['Market', 'Order Region', 'Shipping Mode', 'Category Name', 'Order Country']:
                le = LabelEncoder().fit(df_raw[col].astype(str))
                if base_predict_df[col].iloc[0] in le.classes_:
                     base_predict_df[col] = le.transform(base_predict_df[col])
                else: # Handle unseen category
                     base_predict_df[col] = -1 # Or some other default encoded value
            
            base_prediction = resources['base_model'].predict(base_predict_df[resources['base_model_columns']])

            # Encode Advanced
            adv_predict_df = adv_input_df.copy()
            for col in ['Market', 'Order Region', 'Shipping Mode', 'Category Name', 'Order Country']:
                le = LabelEncoder().fit(df_raw[col].astype(str))
                if adv_predict_df[col].iloc[0] in le.classes_:
                    adv_predict_df[col] = le.transform(adv_predict_df[col])
                else:
                    adv_predict_df[col] = -1
            
            # Encode the new interaction feature
            le_interaction = LabelEncoder().fit(df_raw['Market'].astype(str) + '_' + df_raw['Shipping Mode'].astype(str))
            if adv_predict_df['market_shipping_interaction'].iloc[0] in le_interaction.classes_:
                adv_predict_df['market_shipping_interaction'] = le_interaction.transform(adv_predict_df['market_shipping_interaction'])
            else:
                adv_predict_df['market_shipping_interaction'] = -1

            advanced_prediction = resources['advanced_model'].predict(adv_predict_df[resources['advanced_model_columns']])

            # --- Display Results ---
            st.subheader("🔮 Tahmin Sonuçları")
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.info("📦 **Temel Model (XGBoost)**")
                st.metric("Tahmini Gecikme", f"{int(round(base_prediction[0]))} gün")
                st.write("Sadece temel kategorik verileri kullanır.")
            with res_col2:
                st.success("✨ **Gelişmiş Model (LightGBM)**")
                st.metric("Tahmini Gecikme", f"{int(round(advanced_prediction[0]))} gün")
                st.write("Yeni özellikler ve aykırı değer optimizasyonu içerir.")
        except Exception as e:
            st.error(f"Tahmin sırasında bir hata oluştu: {e}")


with tab2:
    st.header("📈 Modellerin Test Performansı Karşılaştırması")
    st.write("""
    Bu grafik, iki modelin de daha önce görmediği test verileri üzerindeki performansını göstermektedir. 
    Metriklerin ne anlama geldiğini ve hangi modelin daha başarılı olduğunu buradan anlayabilirsiniz.
    """)

    # --- Create Comparison Chart ---
    metrics_df = pd.DataFrame({
        'Temel Model': resources['base_model_metrics'],
        'Gelişmiş Model': resources['advanced_model_metrics']
    })
    
    # R-squared chart
    st.subheader("R-squared ($R^2$) Değerleri")
    st.write("*(Daha yüksek daha iyi)*")
    r2_fig = go.Figure(data=[
        go.Bar(name='Temel Model', x=['R-squared'], y=[metrics_df['Temel Model']['R-squared']]),
        go.Bar(name='Gelişmiş Model', x=['R-squared'], y=[metrics_df['Gelişmiş Model']['R-squared']])
    ])
    r2_fig.update_layout(yaxis_title="R² Skoru", barmode='group')
    st.plotly_chart(r2_fig, use_container_width=True)

    # Error charts (MAE & RMSE)
    st.subheader("Hata Metrikleri (MAE & RMSE)")
    st.write("*(Daha düşük daha iyi)*")
    error_metrics = ['Mean Absolute Error (MAE)', 'Root Mean Squared Error (RMSE)']
    error_fig = go.Figure(data=[
        go.Bar(name='Temel Model', x=error_metrics, y=[metrics_df['Temel Model'][m] for m in error_metrics]),
        go.Bar(name='Gelişmiş Model', x=error_metrics, y=[metrics_df['Gelişmiş Model'][m] for m in error_metrics])
    ])
    error_fig.update_layout(yaxis_title="Gecikme Günü", barmode='group')
    st.plotly_chart(error_fig, use_container_width=True)