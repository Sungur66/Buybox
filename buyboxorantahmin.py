import streamlit as st
import pandas as pd
import joblib

# Model ve encoder yükleme
loaded_model = joblib.load('AmazonBuyBox_model.pkl')

# CSS ekleme
st.markdown(
    """
    <style>
    .main {
        background-color: orange;
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
st.markdown('<h1 class="title">Amazon buybox Tahmini yapmak</h1>', unsafe_allow_html=True)

# Kullanıcıdan veri alma
st.markdown('<h2 class="header">Yeni Veri Girişi yapiniz</h2>', unsafe_allow_html=True)
price = st.number_input('1- Price', value=0.0)
comparison_price_type = st.selectbox('2- Comparison Price Type', ["Was Price", "Other"])
buy_box_item_price = st.number_input('3- Buy Box Item Price', value=0.0)
ship_methods = st.selectbox('4- Ship Methods', ["ALWAYS_TWO_DAYS", "NETWORK_GEO", "OTHER"])
shipping_weight = st.number_input('5- Shipping Weight', value=0.0)
fulfillment_type = st.selectbox('6- Fulfillment Type', ["WFS Eligible", "Walmart Fulfilled", "Seller Fulfilled", "Other"])
average_rating = st.number_input('7- Average Rating', value=0.0)
cost = st.number_input('8- Cost', value=0.0)

# Veriyi DataFrame'e çevirme
new_data = pd.DataFrame({
    "Price": [price],
    "Comparison_Price_Type": [comparison_price_type],
    "Buy_Box_Item_Price": [buy_box_item_price],
    "Ship_Methods": [ship_methods],
    "Shipping_Weight": [shipping_weight],
    "Fulfillment_Type": [fulfillment_type],
    "Average_Rating": [average_rating],
    "Cost": [cost]
})

# Tahmin yapma
if st.button('Buy Box icin Fiyat uygunlugu Tahmini Yap'):
    predictions = loaded_model.predict(new_data)
    predictions_proba = loaded_model.predict_proba(new_data)
    st.write('Buy Box Tahmini:', predictions[0])
    st.write('Tahminin Olasılıkları:', predictions_proba)
