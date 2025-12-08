import pandas as pd
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# --- Chemins de Fichiers ---
INPUT_TRAIN = os.path.join('data', 'fashion-mnist', 'fashion-mnist_train.csv')
INPUT_TEST = os.path.join('data', 'fashion-mnist', 'fashion-mnist_test.csv')
MODEL_OUTPUT = os.path.join('models', 'fashion_classifier.keras')

def train_model(train_path, test_path, model_output_path):
    """
    Charge les données, entraîne un réseau de neurones simple, et sauvegarde le modèle.
    """
    print("--- Démarrage de l'entraînement du modèle ---")

    # 1. Chargement des données prétraitées
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    # Séparation des features (X) et des labels (y)
    X_train = df_train.drop('label', axis=1).values
    y_train = df_train['label'].values
    X_test = df_test.drop('label', axis=1).values
    y_test = df_test['label'].values
    
    # Le dataset est déjà normalisé (si vous l'aviez fait manuellement, sinon ajouter ici une étape de normalisation si nécessaire)
    # Assurez-vous que les valeurs sont entre 0 et 1.
    
    # 2. Encodage One-Hot des labels (nécessaire pour la classification multi-classes dans Keras)
    num_classes = 10
    y_train_encoded = to_categorical(y_train, num_classes=num_classes)
    y_test_encoded = to_categorical(y_test, num_classes=num_classes)

    # 3. Définition du modèle (Réseau de Neurones Simple)
    model = Sequential([
        Dense(512, activation='relu', input_shape=(784,)),
        Dense(256, activation='relu'),
        Dense(num_classes, activation='softmax') # La couche de sortie utilise softmax
    ])

    # 4. Compilation et Entraînement
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

    # Entraînement sur un petit nombre d'epochs pour la rapidité
    model.fit(X_train, y_train_encoded, epochs=5, batch_size=256, verbose=1)

    # 5. Création du dossier de sortie et Sauvegarde du modèle
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    model.save(model_output_path)

    print(f"Modèle sauvegardé avec succès à : {model_output_path}")
    print("--- Entraînement terminé ---")

if __name__ == "__main__":
    train_model(INPUT_TRAIN, INPUT_TEST, MODEL_OUTPUT)