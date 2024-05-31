from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Modeli yükleme
model = load_model('trained_model8480.h5')

# Sınıf isimleri
class_names = ['Clothes', 'Groceries', 'Health', 'Home', 'Kitchen', 'Office', 'Pet Supplies', 'Sports', 'Tools']

def prepare_image(img, target_size=(224, 224)):
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalization
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    img = Image.open(io.BytesIO(file.read()))
    img_array = prepare_image(img)
    
    predictions = model.predict(img_array)
    predicted_probabilities = predictions[0] * 100
    predicted_class = np.argmax(predicted_probabilities)
    predicted_percentage = predicted_probabilities[predicted_class]

    result = {class_names[i]: float(predicted_probabilities[i]) for i in range(len(class_names))}
    result['predicted_class'] = class_names[predicted_class]
    result['predicted_percentage'] = float(predicted_percentage)

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
