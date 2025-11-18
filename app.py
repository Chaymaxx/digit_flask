from flask import Flask, render_template, request, jsonify
import numpy as np
from PIL import Image
import io
import base64
from model import load_model, predict_digit

app = Flask(__name__)

# Charger le modèle au démarrage
model = load_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Récupérer l'image depuis la requête
        if 'image' not in request.files:
            return jsonify({'error': 'Aucune image fournie'}), 400
        
        image_file = request.files['image']
        
        # Lire l'image
        image = Image.open(image_file.stream)
        
        # Faire la prédiction
        prediction, confidence = predict_digit(model, image)
        
        return jsonify({
            'prediction': int(prediction),
            'confidence': float(confidence)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)