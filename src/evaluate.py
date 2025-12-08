import pandas as pd
import os
import json
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, f1_score

# --- Chemins de Fichiers ---
MODEL_INPUT = os.path.join('models', 'fashion_classifier.keras')
INPUT_TEST = os.path.join('data', 'fashion-mnist', 'fashion-mnist_test.csv')
METRICS_OUTPUT = os.path.join('metrics', 'metrics.json')

def evaluate_model(model_path, test_path, metrics_output_path):
    """
    Charge le modèle, évalue sa performance sur les données de test, 
    et sauvegarde les métriques au format JSON pour DVC.
    """
    print("--- Démarrage de l'évaluation du modèle ---")

    # 1. Chargement des données de test
    df_test = pd.read_csv(test_path)
    X_test = df_test.drop('label', axis=1).values
    y_test = df_test['label'].values
    
    # 2. Chargement du modèle
    # Keras load_model lit le modèle depuis models/fashion_classifier.keras
    model = load_model(model_path)
    
    # 3. Prédictions
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # 4. Calcul des métriques
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    # 5. Sauvegarde des métriques au format JSON (pour le tracking DVC)
    metrics = {
        'accuracy': acc,
        'f1_score': f1,
        'num_classes': 10
    }
    
    os.makedirs(os.path.dirname(metrics_output_path), exist_ok=True)
    with open(metrics_output_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"Métriques (Accuracy: {acc:.4f}, F1-Score: {f1:.4f}) sauvegardées à: {metrics_output_path}")
    print("--- Évaluation terminée ---")

if __name__ == "__main__":
    evaluate_model(MODEL_INPUT, INPUT_TEST, METRICS_OUTPUT)