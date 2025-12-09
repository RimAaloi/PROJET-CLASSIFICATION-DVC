# 1. Utiliser une image Python officielle légère (slim) comme environnement de base
FROM python:3.11-slim

# 2. Définir le répertoire de travail dans le conteneur
WORKDIR /app

# 3. Copier le fichier des dépendances et installer toutes les librairies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --timeout 1000

# 4. Copier le reste du code source et les fichiers de configuration DVC
COPY . .

# 5. Définir les identifiants AWS comme variables d'environnement (CRUCIAL pour DVC S3)
# Les valeurs seront injectées SÉCURISÉMENT lors de l'exécution (docker run -e...).
ENV AWS_ACCESS_KEY_ID=""
ENV AWS_SECRET_ACCESS_KEY=""

# 6. Définir le répertoire de données comme volume (bonne pratique DVC)
VOLUME /app/data

# 7. Définir DVC comme point d'entrée pour toutes les commandes.
ENTRYPOINT ["dvc"]

# 8. Commande par défaut : exécuter 'repro'. Si l'utilisateur tape 'docker run <image> pull',
# cela devient 'dvc pull'.
CMD ["repro"]