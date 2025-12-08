# 1. Utiliser une image Python officielle légère (slim) comme environnement de base
FROM python:3.11-slim

# 2. Définir le répertoire de travail dans le conteneur
WORKDIR /app

# 3. Copier le fichier des dépendances et installer toutes les librairies
# Ce processus installe DVC, dvc-s3, TensorFlow, scikit-learn, etc., à partir de votre liste figée.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copier le reste du code source et les fichiers de configuration DVC
# Cela inclut src/, dvc.yaml, .dvc/config et les pointeurs de données (.dvc)
COPY . .

# 5. Définir les identifiants AWS comme variables d'environnement (CRUCIAL pour DVC S3)
# Ces variables permettent à DVC (via la librairie boto3) d'accéder à votre bucket S3.
# Optionnel : définir dvc comme point d'entrée pour lancer les commandes facilement
ENTRYPOINT ["dvc"]