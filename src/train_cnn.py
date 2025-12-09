import pandas as pd
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# --- Chemins de Fichiers ---
INPUT_TRAIN = os.path.join('data', 'fashion-mnist', 'fashion-mnist_train.csv')
MODEL_OUTPUT = os.path.join('models', 'cnn_model.keras')
IMAGE_SIZE = 28
NUM_CLASSES = 10

def train_cnn(train_path, model_output_path):
    """
    Entraîne un modèle CNN simple pour la classification d'images Fashion-MNIST.
    """
    print("--- Démarrage de l'entraînement du modèle CNN ---")

    # 1. Chargement et préparation des données
    df_train = pd.read_csv(train_path)
    X = df_train.drop('label', axis=1).values
    y = df_train['label'].values
    
    # 2. Reshape pour le CNN: (nombre d'images, hauteur, largeur, canaux)
    # Les images sont en niveaux de gris (1 canal)
    X_reshaped = X.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
    y_encoded = to_categorical(y, num_classes=NUM_CLASSES)
    
    # Séparation train/validation pour vérifier la sur-apprentissage
    X_train, X_val, y_train, y_val = train_test_split(X_reshaped, y_encoded, test_size=0.1, random_state=42)

    # 3. Architecture CNN
    model = Sequential([
        # Couche de Convolution 1
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1)),
        MaxPooling2D((2, 2)),
        
        # Couche de Convolution 2
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        # 4. Classification (Dense)
        Flatten(),
        Dense(100, activation='relu'),
        Dense(NUM_CLASSES, activation='softmax')
    ])

    # 5. Compilation et Entraînement
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_val, y_val), verbose=1)

    # 6. Sauvegarde du modèle
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    model.save(model_output_path)

    print(f"Modèle CNN sauvegardé avec succès à : {model_output_path}")
    print("--- Entraînement CNN terminé ---")

if __name__ == "__main__":
    train_cnn(INPUT_TRAIN, MODEL_OUTPUT)