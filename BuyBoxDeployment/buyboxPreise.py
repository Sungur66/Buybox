#  Buy Box Fiyat Tahmini Streamlit

import streamlit as st
import pandas as pd
import joblib

# Model ve scaler yükleme
loaded_model = joblib.load('BuyBoxPrice_model.pkl')
loaded_scaler = joblib.load('BuyBoxPriceScaler.pkl')


# CSS ekleme
st.markdown(
    """
    <style>
    .main {
        background-color: #e6f0ff;
    }
    .title {
        color: #003366;
        text-align: center;
    }
    .header {
        color: #004080;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Başlık
st.markdown('<h1 class="title">Buy Box Fiyat Tahmini</h1>', unsafe_allow_html=True)

# Kullanıcıdan veri alma
st.markdown('<h2 class="header">Yeni Veri Girişi yapınız</h2>', unsafe_allow_html=True)
price = st.number_input('1- Price', value=0.0)
comparison_price = st.number_input('2- Comparison Price', value=0.0)
msrp = st.number_input('3- MSRP', value=0.0)
shipping_weight = st.number_input('4- Shipping Weight', value=0.0)
average_rating = st.number_input('5- Average Rating', value=0.0)

# Veriyi DataFrame'e çevirme
new_data = pd.DataFrame({
    "Price": [price],
    "Comparison_Price": [comparison_price],
    "MSRP": [msrp],
    "Shipping_Weight": [shipping_weight],
    "Average_Rating": [average_rating]
})

# Tahmin yapma
if st.button('Buy Box Fiyat Tahmini Yap'):
    urunler_scaled = loaded_scaler.transform(new_data)
    predictions = loaded_model.predict(urunler_scaled)
    st.write('Buy Box Fiyat Tahmini:', predictions[0])
