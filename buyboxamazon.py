import streamlit as st
import pandas as pd
import joblib

# Modeli yükle
loaded_model = joblib.load('AmazonBuyBox_model.pkl')

# Streamlit başlığı
st.title('Amazon Buy Box Prediction')

# Kullanıcı girdileri için form oluştur
st.header('Yeni Verileri Girin')

# Kullanıcıdan giriş al
new_offer_count = st.number_input('1- New Offer Count: 180 days avg.', value=0.0)
sessions_total = st.number_input('2- Sessions - Total', value=0)
page_views_total = st.number_input('3- Page Views - Total', value=0)
winner_count = st.number_input('4- Buy Box: Winner Count 365 days', value=0.0)
top_seller = st.number_input('5- Buy Box: % Top Seller 365 days', value=0.0)
unit_session_percentage = st.number_input('6- Unit Session Percentage', value=0.0)
units_ordered = st.number_input('7- Units Ordered', value=0)
oos_90_days = st.number_input('8- Buy Box ??: 90 days OOS', value=0.0)
rating = st.number_input('9- Reviews: Rating', value=0.0)
sales_rank_drops = st.number_input('10- Sales Rank: Drops last 180 days', value=0.0)
avg_180_days = st.number_input('11- Buy Box ??: 180 days avg.', value=0.0)
amazon_365_days = st.number_input('12- Buy Box: % Amazon 365 days', value=0.0)

# Tahmin butonu
if st.button('Tahmin Et'):
    # Kullanıcı girdilerini DataFrame'e çevir
    new_data = pd.DataFrame({
        'New Offer Count: 180 days avg.': [new_offer_count],
        'Sessions - Total': [sessions_total],
        'Page Views - Total': [page_views_total],
        'Buy Box: Winner Count 365 days': [winner_count],
        'Buy Box: % Top Seller 365 days': [top_seller],
        'Unit Session Percentage': [unit_session_percentage],
        'Units Ordered': [units_ordered],
        'Buy Box ??: 90 days OOS': [oos_90_days],
        'Reviews: Rating': [rating],
        'Sales Rank: Drops last 180 days': [sales_rank_drops],
        'Buy Box ??: 180 days avg.': [avg_180_days],
        'Buy Box: % Amazon 365 days': [amazon_365_days]
    })

    # Tahmin ve olasılık hesaplama
    predictions = loaded_model.predict(new_data)
    predictions_proba = loaded_model.predict_proba(new_data)

    # Sonuçları göster
    st.subheader('Tahmin Sonucu')
    st.write('Tahmin:', predictions[0])
    
    st.subheader('Tahmin Olasılıkları')
    st.write('Olasılıklar:', predictions_proba[0])

