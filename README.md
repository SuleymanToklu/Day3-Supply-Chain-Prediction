# 🚀 Proje Hyperion: Dinamik Tedarik Zinciri Risk Tahmin Sistemi

Bu proje, AI Maratonu'mun 3. gününde hayata geçirilmiştir. Basit bir gecikme tahmin modelinden, dış dünya verileriyle (hava durumu, tatiller) zenginleştirilmiş, etkileşimli bir risk analiz platformuna dönüştürülmüştür.

### ✨ Öne Çıkan Özellikler

- **Çoklu Veri Entegrasyonu:** API'ler aracılığıyla canlı hava durumu ve küresel tatil verilerini entegre eder.
- **Akıllı Özellik Mühendisliği:** Ham veriden, modelin performansını artıran anlamlı özellikler (`is_holiday`, `max_temp` vb.) üretir.
- **Gelişmiş Modelleme:** LightGBM kullanarak aykırı değerlere karşı dayanıklı ve isabetli bir tahmin modeli sunar.
- **Etkileşimli Komuta Merkezi:** Streamlit ile geliştirilen modern arayüz, anlık olarak geleceğe dönük tahminler yapmaya olanak tanır.

### 💻 Kullanılan Teknolojiler
- Python, Pandas, NumPy
- Scikit-learn, LightGBM
- Streamlit, Plotly
- `requests`, `holidays`, `yfinance`
- Git & GitHub

### 🚀 Nasıl Çalıştırılır?
1.  Repo'yu klonla: `git clone ...`
2.  Gerekli kütüphaneleri yükle: `pip install -r requirements.txt`
3.  Zenginleştirilmiş veri setini oluştur (API'lerden veri çeker): `python build_features.py`
4.  Modeli eğit: `python train_model.py`
5.  Uygulamayı başlat: `streamlit run app.py`
