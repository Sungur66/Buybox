import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Modeli yükleme
model = load_model('trained_model8480.h5')

# Sınıf isimleri (kendi sınıf isimlerinizle değiştirmelisiniz)
class_names = ['Clothes', 'Groceries', 'Health', 'Home', 'Kitchen', 'Office', 'Pet Supplies', 'Sports', 'Tools']

# Resmi yükleme ve hazırlama fonksiyonu
def prepare_image(img, target_size=(224, 224)):
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalization
    return img_array

# Streamlit arayüzü
st.title("Image Classification with TensorFlow")
st.header("Upload an image to classify")

# Kullanıcıdan resim yüklemesini isteme
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Resmi görüntüleme
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Resmi hazırlama ve tahmin yapma
    img_array = prepare_image(img)
    predictions = model.predict(img_array)

    # Tahmin sonuçlarını yorumlama
    predicted_probabilities = predictions[0] * 100  # Yüzdeye çevirmek için 100 ile çarpıyoruz
    predicted_class = np.argmax(predicted_probabilities)
    predicted_percentage = predicted_probabilities[predicted_class]

    # Sonuçları gösterme
    for i, class_name in enumerate(class_names):
        st.write(f"{class_name}: {predicted_probabilities[i]:.2f}%")

    st.write(f'Tahmin edilen sınıf: **{class_names[predicted_class]}** ({predicted_percentage:.2f}%)')
