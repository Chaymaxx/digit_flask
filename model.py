import tensorflow as tf
import numpy as np
from PIL import Image

def load_model():
    """Charge le modèle pré-entraîné"""

    model = tf.keras.models.load_model('models/mnist_cnn_model.h5')
    print("Modèle chargé avec succès!")
    return model


def preprocess_image(image):
    """
    Prétraite l'image pour la prédiction
    - Convertit en niveaux de gris
    - Redimensionne à 28x28
    - Normalise les valeurs
    """
    # Convertir en niveaux de gris
    image = image.convert('L')
    
    # Redimensionner à 28x28
    image = image.resize((28, 28))
    
    # Convertir en array numpy
    img_array = np.array(image)
    
    # Inverser les couleurs si nécessaire (fond blanc -> fond noir)
    # MNIST a un fond noir et des chiffres blancs
    if img_array.mean() > 127:
        img_array = 255 - img_array
    
    # Normaliser entre 0 et 1
    img_array = img_array / 255.0
    
    # Ajouter les dimensions pour le batch et le canal
    img_array = img_array.reshape(1, 28, 28, 1)
    
    return img_array

def predict_digit(model, image):
    """
    Fait une prédiction sur l'image
    Retourne le chiffre prédit et la confiance
    """
    # Prétraiter l'image
    processed_image = preprocess_image(image)
    
    # Faire la prédiction
    predictions = model.predict(processed_image, verbose=0)
    
    # Obtenir le chiffre prédit et la confiance
    predicted_digit = np.argmax(predictions[0])
    confidence = predictions[0][predicted_digit] * 100
    
    return predicted_digit, confidence