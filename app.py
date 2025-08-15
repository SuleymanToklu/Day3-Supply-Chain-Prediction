import streamlit as st
import pandas as pd
import joblib
import json
import requests
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

st.set_page_config(
    page_title="Proje Hyperion",
    page_icon="🚀",
    layout="wide"
)

COUNTRY_DISPLAY_MAP = {
    "ABD": "Estados Unidos",
    "Almanya": "Alemania",
    "Avustralya": "Australia",
    "Fransa": "Francia",
    "İngiltere": "Reino Unido",
    "Meksika": "México",
    "Dominik Cumhuriyeti": "República Dominicana",
    "El Salvador": "El Salvador"
}
@st.cache_resource
def load_resources():
    """Loads the Hyperion model and related artifacts."""
    try:
        model = joblib.load('hyperion_model.pkl')
        columns = joblib.load('hyperion_model_columns.pkl')
        df_raw = pd.read_csv("DataCoSupplyChainDataset.csv", encoding='latin1', low_memory=False)
        return model, columns, df_raw
    except FileNotFoundError:
        return None, None, None

def get_live_weather(lat, lon, date):
    """Fetches weather forecast for a future date."""
    date_str = date.strftime('%Y-%m-%d')
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=temperature_2m_max,precipitation_sum&start_date={date_str}&end_date={date_str}"
    try:
        response = requests.get(url)
        data = response.json()['daily']
        return {'max_temp': data['temperature_2m_max'][0], 'precipitation': data['precipitation_sum'][0]}
    except Exception:
        return {'max_temp': 15, 'precipitation': 0}

model, columns, df_raw = load_resources()
if not model or not columns or df_raw is None:
    st.error("Hyperion modeli veya gerekli dosyalar bulunamadı. Lütfen `train_model.py` script'inin başarıyla çalıştığından emin olun.")
    st.stop()

tab1, tab2 = st.tabs(["🚀 Komuta Merkezi", "📖 Proje Hakkında"])

with tab1:
    with st.sidebar:
        st.title("🛰️ Hyperion Girdi Paneli")
        st.write("Tahmin yapmak için sevkiyat bilgilerini girin.")

        st.subheader("📍 Lokasyon Bilgileri")
        display_countries = sorted(list(COUNTRY_DISPLAY_MAP.keys()))
        selected_display_country = st.selectbox("Sipariş Ülkesi", options=display_countries)
        order_country = COUNTRY_DISPLAY_MAP[selected_display_country] 
        
        market = st.selectbox("Pazar (Market)", options=sorted(df_raw['Market'].unique()))
        
        st.subheader("🚚 Sevkiyat Bilgileri")
        shipping_mode = st.selectbox("Kargo Türü", options=sorted(df_raw['Shipping Mode'].unique()))
        order_date = st.date_input("Sipariş Tarihi", value=datetime.now())

        st.subheader("📦 Ürün Bilgileri")
        category_name = st.selectbox("Ürün Kategorisi", options=sorted(df_raw['Category Name'].unique()))
        sales = st.number_input("Satış Tutarı (Sales)", min_value=0.0, value=250.0, step=50.0)
        quantity = st.number_input("Ürün Adedi", min_value=1, value=3)

        predict_button = st.button("Riski ve Gecikmeyi Tahmin Et", use_container_width=True, type="primary")

    st.title("🚀 Hyperion Tedarik Zinciri Risk Merkezi")

    if predict_button:
        with st.spinner('Canlı veriler alınıyor ve analiz yapılıyor...'):
            country_coords = {
                "Estados Unidos": (38.9, -77.0), "México": (19.4, -99.1), "Francia": (48.8, 2.3),
                "Reino Unido": (51.5, -0.1), "Alemania": (52.5, 13.4), "Australia": ( -33.8, 151.2),
                "República Dominicana": (18.48, -69.93), "El Salvador": (13.70, -89.20)
            }
            lat, lon = country_coords.get(order_country, (38.9, -77.0))
            live_weather = get_live_weather(lat, lon, order_date)
            
            input_data = {
                'Type': 'DEBIT', 'Delivery Status': 'Shipping on time', 'Category Name': category_name,
                'Customer Country': order_country, 'Market': market, 'Order Country': order_country,
                'Order Region': 'West of USA', 'Shipping Mode': shipping_mode, 'Sales': sales,
                'Order Item Quantity': quantity, 'is_holiday': 0, 
                'max_temp': live_weather['max_temp'], 'precipitation': live_weather['precipitation']
            }
            input_df = pd.DataFrame([input_data])

            for col in columns:
                if col in input_df.columns:
                    if col in df_raw.columns and df_raw[col].dtype == 'object':
                        le = LabelEncoder().fit(df_raw[col].astype(str))
                        try: input_df[col] = le.transform(input_df[col])
                        except ValueError: input_df[col] = -1 
            
            input_df = input_df.reindex(columns=columns, fill_value=0)
            prediction = model.predict(input_df)
            delay_days = int(round(prediction[0]))

            st.subheader(f"🗓️ {order_date.strftime('%d %B %Y')} Tarihli Sevkiyat Analizi")
            col1, col2 = st.columns(2)
            with col1:
                with st.container(border=True):
                    risk_level = "Düşük"
                    risk_delta = "Gecikme Beklenmiyor"
                    if 3 < delay_days <= 7: risk_level = "Orta"; risk_delta = "Hafif Gecikme Riski"
                    elif delay_days > 7: risk_level = "⚠️ Yüksek"; risk_delta = "Ciddi Gecikme Riski"
                    st.metric(label="Risk Seviyesi", value=risk_level, delta=risk_delta, delta_color="inverse")
            with col2:
                with st.container(border=True):
                    st.metric(label="Tahmini Gecikme Süresi", value=f"{delay_days} Gün")
                    st.write(f"Beklenen Hava Durumu: {live_weather['max_temp']}°C, {live_weather['precipitation']}mm Yağış")
            st.success("Analiz tamamlandı.")
    else:
        st.info("Lütfen sol paneldeki bilgileri girip tahmin butonuna basın.")

with tab2:
    st.header("📖 Proje Hakkında: Hyperion")
    st.markdown("""
    **Hyperion**, statik bir veri setinin sınırlarını aşarak, bir tedarik zinciri operasyonundaki potansiyel gecikmeleri **gerçek dünya faktörlerini** de hesaba katarak tahmin etmeyi amaçlayan bir makine öğrenmesi projesidir.
    """)

    st.subheader("🚀 Projenin Serüveni")
    st.markdown("""
    Bu proje, basit bir regresyon modellemesi hedefiyle başladı. Ancak, sadece mevcut verilerle yapılan tahminlerin bir noktada tıkandığı ve gerçek dünyadaki dinamikleri yansıtmakta yetersiz kaldığı görüldü. Bu noktada proje, **Hyperion vizyonuyla** yeniden doğdu.
    
    1.  **Veri Zenginleştirme:** Ana veri seti, üç kritik harici veri kaynağı ile birleştirildi:
        * **Küresel Tatil Takvimleri:** Siparişin çıkış/varış ülkesindeki resmi tatillerin gecikmelere etkisi.
        * **Tarihsel Hava Durumu:** Sevkiyat lokasyonlarındaki aşırı hava olaylarının lojistiğe etkisi.
        * **Ekonomik Göstergeler:** Küresel petrol fiyatlarının sevkiyat maliyetlerine ve kararlarına etkisi (Bu özellik geliştirme aşamasında geçici olarak askıya alınmıştır).
    2.  **Veri Mühendisliği:** Bu farklı kaynaklardan gelen veriler, `build_features.py` adlı bir script ile temizlendi, birleştirildi ve `hyperion_dataset.csv` adında modellemeye hazır, zenginleştirilmiş bir veri seti oluşturuldu.
    3.  **Gelişmiş Modelleme:** Bu yeni ve zengin veri seti, aykırı değerlere karşı daha dayanıklı olan **Huber loss** kriterine sahip bir **LightGBM** modeli ile eğitildi.
    4.  **Komuta Merkezi:** Sonuçların sunulması için, anlık olarak geleceğe dönük hava durumu tahmini çekebilen, modern ve etkileşimli bir Streamlit arayüzü tasarlandı.
    """)

    st.subheader("🛠️ Kullanılan Teknolojiler")
    st.markdown("""
    - **Programlama Dili:** Python
    - **Veri Analizi ve Manipülasyon:** Pandas, NumPy
    - **Makine Öğrenmesi:** Scikit-learn, LightGBM
    - **Web Arayüzü ve Dashboard:** Streamlit
    - **Veri Görselleştirme:** Plotly
    - **Harici Veri Çekme:** `requests` (API için) ve `holidays`
    - **Versiyon Kontrolü:** Git & GitHub
    """)

st.sidebar.markdown("---")
st.sidebar.info("Bu proje, Süleyman Toklu tarafından geliştirilmiştir.")