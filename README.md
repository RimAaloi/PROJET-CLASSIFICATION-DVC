#  Projet Classification Fashion-MNIST avec DVC + MLOps

[![MLOps Pipeline](https://github.com/RimAaloi/PROJET-CLASSIFICATION-DVC/actions/workflows/mlops.yml/badge.svg)](https://github.com/RimAaloi/PROJET-CLASSIFICATION-DVC/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![DVC](https://img.shields.io/badge/dvc-3.64.2-blue.svg)](https://dvc.org/)
[![TensorFlow](https://img.shields.io/badge/tensorflow-2.x-orange.svg)](https://www.tensorflow.org/)

##  Table des matières

- [Description du projet](#description-du-projet)
- [Architecture et pipeline](#architecture-et-pipeline)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Structure du projet](#structure-du-projet)
- [Modèles d'apprentissage](#modèles-dapprentissage)
- [Méthodologie DVC](#méthodologie-dvc)
- [Pipeline MLOps](#pipeline-mlops)
- [Résultats et métriques](#résultats-et-métriques)
- [Contribution](#contribution)

---

##  Description du projet

Ce projet met en œuvre une **pipeline MLOps complète** pour la classification d'images du dataset **Fashion-MNIST** en utilisant trois approches différentes :

1. **Réseau de neurones simple (MLP)** - Baseline rapide
2. **Réseau de neurones convolutifs (CNN)** - Modèle optimisé
3. **Transfer Learning (MobileNetV2)** - Modèle pré-entraîné

Le projet utilise **DVC (Data Version Control)** pour gérer les données, les modèles et les artefacts, et **GitHub Actions + CML** pour automatiser et rapporter les résultats du pipeline.

###  Dataset
- **Dataset** : Fashion-MNIST (60,000 images d'entraînement, 10,000 de test)
- **Classes** : 10 (T-shirt, Pantalon, Pull, Robe, Manteau, Sandal, Chemise, Sneaker, Sac, Botte)
- **Format d'image** : 28×28 pixels en niveaux de gris

---

##  Architecture et pipeline

### Vue d'ensemble
```
Data (Fashion-MNIST)
    ↓
[Train Simple MLP] → models/fashion_classifier.keras
[Train CNN]        → models/cnn_model.keras
[Train Transfer]   → models/transfer_model.keras
    ↓
[Evaluate] → metrics/metrics.json + plots/
    ↓
[CML Report] → Commentaire PR sur GitHub
```

### Étapes du pipeline DVC

| Étape | Entrée | Sortie | Description |
|-------|--------|--------|-------------|
| **train_simple** | CSV d'entraînement | `fashion_classifier.keras` | Entraîne un MLP simple |
| **train_cnn** | CSV d'entraînement | `cnn_model.keras` | Entraîne un CNN 2D |
| **train_transfer** | CSV d'entraînement | `transfer_model.keras` | Entraîne MobileNetV2 fine-tuné |
| **evaluate** | Tous les modèles + CSV test | `metrics.json` + graphiques | Évalue et compare les 3 modèles |

---

##  Installation

### Prérequis
- Python 3.11+
- Git
- pip ou conda

### 1. Cloner le repository
```bash
git clone https://github.com/RimAaloi/PROJET-CLASSIFICATION-DVC.git
cd PROJET-CLASSIFICATION-DVC
```

### 2. Créer un environnement virtuel
```bash
python -m venv .venv

# Sur Windows
.\.venv\Scripts\activate

# Sur macOS/Linux
source .venv/bin/activate
```

### 3. Installer les dépendances
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configurer DVC (optionnel - pour le stockage distant)
```bash
# Si vous avez un stockage S3
dvc remote add -d s3_remote s3://bucket-name/path
dvc remote modify s3_remote access_key_id <YOUR_AWS_KEY>
dvc remote modify s3_remote secret_access_key <YOUR_AWS_SECRET>
```

### 5. Récupérer les données et modèles
```bash
dvc pull
```

---

##  Utilisation

### Option 1 : Exécuter l'intégralité du pipeline
```bash
dvc repro
```

Cela va :
1. Entraîner le MLP simple
2. Entraîner le CNN
3. Entraîner le modèle Transfer Learning
4. Évaluer les 3 modèles et générer les métriques

### Option 2 : Entraîner un modèle spécifique
```bash
# Simple MLP
python src/train.py

# CNN
python src/train_cnn.py

# Transfer Learning
python src/train_transfer.py
```

### Option 3 : Évaluer les modèles
```bash
python src/evaluate.py
```

Cela génère :
- `metrics/metrics.json` - Accuracy et F1-score pour chaque modèle
- `metrics/plots/confusion_matrix_*.png` - Matrices de confusion

### Pousser les changements
```bash
git add .
git commit -m "Train models and evaluate results"
git push

# Le workflow GitHub Actions se déclenche automatiquement
# Consultez l'onglet "Actions" pour suivre l'exécution
```

---

##  Structure du projet

```
PROJET-CLASSIFICATION-DVC/
├── 📄 README.md                          # Ce fichier
├── 📄 requirements.txt                   # Dépendances Python
├── 📄 dvc.yaml                           # Configuration du pipeline DVC
├── 📄 data.dvc                           # Référence DVC pour les données
├── 🐳 Dockerfile                         # Image Docker pour le projet
│
├── 📁 data/
│   └── 📁 fashion-mnist/
│       ├── fashion-mnist_train.csv       # 60,000 images d'entraînement
│       ├── fashion-mnist_test.csv        # 10,000 images de test
│       └── [fichiers binaires MNIST]
│
├── 📁 src/
│   ├── 🐍 train.py                       # Entraînement MLP simple
│   ├── 🐍 train_cnn.py                   # Entraînement CNN
│   ├── 🐍 train_transfer.py              # Entraînement Transfer Learning
│   └── 🐍 evaluate.py                    # Évaluation des 3 modèles
│
├── 📁 models/
│   ├── fashion_classifier.keras          # Modèle MLP entraîné
│   ├── cnn_model.keras                   # Modèle CNN entraîné
│   └── transfer_model.keras              # Modèle Transfer Learning entraîné
│
├── 📁 metrics/
│   ├── metrics.json                      # Accuracy & F1-score
│   └── 📁 plots/
│       ├── confusion_matrix_simple_mlp.png
│       ├── confusion_matrix_cnn.png
│       └── confusion_matrix_transfer_learning.png
│
└── 📁 .github/
    └── 📁 workflows/
        └── mlops.yml                     # Pipeline GitHub Actions
```

---

##  Modèles d'apprentissage

### 1. **Simple MLP (Multi-Layer Perceptron)**
**Fichier** : `src/train.py`

```python
Model: Sequential
├── Dense(128, activation='relu') 
├── Dropout(0.2)
├── Dense(64, activation='relu')
├── Dropout(0.2)
└── Dense(10, activation='softmax')  # 10 classes
```

**Caractéristiques** :
- Rapide à entraîner
- Baseline de comparaison
- ~95% d'accuracy
- Temps d'entraînement : < 1 minute

---

### 2. **CNN (Convolutional Neural Network)**
**Fichier** : `src/train_cnn.py`

```python
Model: Sequential
├── Conv2D(32, 3×3, activation='relu') → MaxPooling2D(2×2)
├── Conv2D(64, 3×3, activation='relu') → MaxPooling2D(2×2)
├── Flatten()
├── Dense(128, activation='relu') → Dropout(0.5)
└── Dense(10, activation='softmax')
```

**Caractéristiques** :
- Exploite les patterns spatiaux des images
- Meilleure performance que MLP
- ~97% d'accuracy
- Temps d'entraînement : 2-3 minutes

---

### 3. **Transfer Learning (MobileNetV2)**
**Fichier** : `src/train_transfer.py`

```python
Model: MobileNetV2 (pré-entraîné)
├── MobileNetV2 (ImageNet weights)
├── Global Average Pooling
├── Dense(256, activation='relu') → Dropout(0.5)
└── Dense(10, activation='softmax')
```

**Caractéristiques** :
- Utilise les poids pré-entraînés sur ImageNet
- Fine-tuning sur Fashion-MNIST
- Meilleure accuracy
- ~98% d'accuracy
- Temps d'entraînement : 3-5 minutes

---

##  Méthodologie DVC

### Qu'est-ce que DVC ?
DVC (Data Version Control) permet de :
- ✅ Versionner les données (comme Git pour les fichiers binaires volumineux)
- ✅ Tracker les modèles et artefacts
- ✅ Automatiser les pipelines ML
- ✅ Gérer le stockage distant (S3, GCS, etc.)

### Configuration DVC (`dvc.yaml`)

```yaml
stages:
  train_simple:
    deps: [src/train.py, data/fashion-mnist/...csv]
    cmd: python src/train.py
    outs: [models/fashion_classifier.keras]
    
  train_cnn:
    deps: [src/train_cnn.py, data/fashion-mnist/...csv]
    cmd: python src/train_cnn.py
    outs: [models/cnn_model.keras]
    
  train_transfer:
    deps: [src/train_transfer.py, data/fashion-mnist/...csv]
    cmd: python src/train_transfer.py
    outs: [models/transfer_model.keras]
    
  evaluate:
    deps: [src/evaluate.py, models/*.keras, data/...]
    cmd: python src/evaluate.py
    metrics:
      - metrics/metrics.json: {cache: false}
    plots:
      - metrics/plots/confusion_matrix_*.png: {cache: false}
```

### Commandes DVC principales

```bash
# Exécuter le pipeline complet
dvc repro

# Voir l'état du pipeline
dvc dag

# Voir les différences entre les versions
dvc plots diff

# Pousser les artefacts vers le stockage distant
dvc push

# Récupérer les artefacts
dvc pull
```

---

##  Pipeline MLOps (GitHub Actions + CML)

### Automatisation avec GitHub Actions

Le fichier `.github/workflows/mlops.yml` automatise :

1. **Checkout** du code
2. **Installation** des dépendances
3. **Récupération** des données via DVC
4. **Exécution** du pipeline DVC
5. **Génération** du rapport avec CML
6. **Publication** des résultats en commentaire PR

### Résultats automatiques

Après chaque `git push`, un commentaire est ajouté à votre PR contenant :

```markdown
##  Rapport d'exécution du pipeline MLOps

| Modèle | Accuracy | F1-score |
|--------|----------|----------|
| Simple MLP | 95.2% | 0.952 |
| CNN | 97.1% | 0.971 |
| Transfer Learning | 98.5% | 0.985 |

###  Matrices de confusion
[Images des matrices de confusion]
```

---

##  Résultats et métriques

### Métriques JSON (`metrics/metrics.json`)

```json
{
  "simple_mlp": {
    "accuracy": 0.952,
    "f1_score": 0.952
  },
  "cnn": {
    "accuracy": 0.971,
    "f1_score": 0.971
  },
  "transfer_learning": {
    "accuracy": 0.985,
    "f1_score": 0.985
  }
}
```

### Interprétation des résultats

| Métrique | Signification |
|----------|--------------|
| **Accuracy** | Pourcentage de prédictions correctes |
| **F1-score** | Moyenne harmonique entre precision et recall |
| **Confusion Matrix** | Détail des erreurs par classe |

### Comment améliorer les résultats ?

1. **Augmentation des données** (data augmentation)
2. **Fine-tuning du transfer learning**
3. **Hyperparameter tuning** (learning rate, batch size)
4. **Ensemble methods** (combiner les 3 modèles)

---

##  Utilisation avec Docker

### Construire l'image
```bash
docker build -t fashion-classifier .
```

### Exécuter le pipeline dans Docker
```bash
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/metrics:/app/metrics \
  fashion-classifier dvc repro
```

---

##  Secrets GitHub Actions

Pour que le pipeline fonctionne avec le stockage S3, ajoutez les secrets :

1. Allez sur **Settings** → **Secrets and variables** → **Actions**
2. Ajoutez :
   - `AWS_ACCESS_KEY_ID`
   - `AWS_SECRET_ACCESS_KEY`

Le `GITHUB_TOKEN` est fourni automatiquement par GitHub.

---

##  Dépendances principales

| Package | Version | Usage |
|---------|---------|-------|
| **TensorFlow** | 2.x | Framework deep learning |
| **DVC** | 3.64.2 | Version control des données |
| **pandas** | Latest | Traitement des données CSV |
| **scikit-learn** | Latest | Métriques (accuracy, F1) |
| **matplotlib/seaborn** | Latest | Visualisations |
| **CML** | Latest | Rapports automatiques |

---

##  Contribution

Pour contribuer au projet :

1. Créez une branche (`git checkout -b feature/ma-feature`)
2. Commitez vos changements (`git commit -m 'Add feature'`)
3. Poussez votre branche (`git push origin feature/ma-feature`)
4. Ouvrez une Pull Request

**Important** : Le pipeline MLOps s'exécutera automatiquement, et les résultats s'afficheront en commentaire PR.

---

##  Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

---

##  Auteur

**Rim Aaloi**
- GitHub : [@RimAaloi](https://github.com/RimAaloi)
- Repository : [PROJET-CLASSIFICATION-DVC](https://github.com/RimAaloi/PROJET-CLASSIFICATION-DVC)

---

##  Support

Pour toute question ou problème :
- Ouvrez une [Issue](https://github.com/RimAaloi/PROJET-CLASSIFICATION-DVC/issues)
- Consultez les [Discussions](https://github.com/RimAaloi/PROJET-CLASSIFICATION-DVC/discussions)

---

##  Ressources d'apprentissage

- [DVC Documentation](https://dvc.org/doc)
- [CML Documentation](https://cml.dev/)
- [TensorFlow Guide](https://www.tensorflow.org/guide)
- [Fashion-MNIST Dataset](https://github.com/zalandoresearch/fashion-mnist)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)

---

**Dernière mise à jour** : Décembre 2025  
**Statut du projet** : ✅ Actif en développement 
## test 
