# 3️⃣ PROJET: TPs-Machine-learning

## 📋 Informations Actuelles
- **Lien**: https://github.com/taibi1995/TPs-Machine-learning
- **Type**: Travaux Pratiques / Education
- **Langage**: Python (100%)
- **Fichiers**: 9 fichiers (TD1-TD9)

## ⚠️ Problèmes Identifiés
1. **README vide ou minimal**
2. **Pas de requirements.txt**
3. **Pas de .gitignore**
4. **Pas de description**
5. **Pas de topics**
6. **Pas de documentation**

---

## 📝 Fichiers à Créer/Modifier

### 1️⃣ README.md (À créer/remplacer)

```markdown
# Travaux Pratiques - Machine Learning 🤖

## 📚 Description
Collection de travaux pratiques couvrant les concepts et techniques fondamentales du Machine Learning, avec des implémentations pratiques et des cas d'usage réels.

## 🎯 Objectifs
- Maîtriser les algorithmes de Machine Learning classiques
- Comprendre le preprocessing et la feature engineering
- Implémenter la validation croisée et l'évaluation de modèles
- Résoudre des problèmes de classification et régression
- Développer l'intuition ML à travers la pratique

## 📋 Liste des Travaux Pratiques

| TD | Sujet | Algorithmes/Concepts |
|----|-------|---------------------|
| **TD1** | Preprocessing | Normalisation, encoding, handling missing values |
| **TD2** | Régression Linéaire | Linear Regression, Gradient Descent |
| **TD3** | Régression Logistique | Binary Classification, Logistic Regression |
| **TD4** | [À compléter] | [À compléter] |
| **TD5** | Classification | Decision Trees, Feature Selection |
| **TD6** | Ensemble Methods | Random Forest, Bagging, Boosting |
| **TD7** | SVM | Support Vector Machines, Kernel Methods |
| **TD8** | Clustering | K-Means, Hierarchical Clustering, DBSCAN |
| **TD9** | Unsupervised Learning | PCA, Dimensionality Reduction |

## 🛠️ Technologies

- **Python 3.8+**
- **Scikit-learn** - Algorithms ML
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Matplotlib / Seaborn** - Visualizations
- **Jupyter Notebook** - Interactive development

## 📥 Installation

```bash
# Cloner le repository
git clone https://github.com/taibi1995/TPs-Machine-learning.git
cd TPs-Machine-learning

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt
```

## 📖 Utilisation

### Exécuter un TP spécifique

```bash
python TD1.py
python td2.py
python td3.py
# etc...
```

### Utiliser Jupyter Notebook

```bash
# Lancer Jupyter
jupyter notebook

# Ouvrir le fichier souhaité
```

## 📂 Structure du Projet

```
.
├── TD1.py                  # Preprocessing et Data Cleaning
├── td2.py                  # Régression Linéaire
├── td3.py                  # Régression Logistique
├── td4.py                  # [À décrire]
├── TD5.py                  # Classification
├── td6.py                  # Ensemble Methods
├── TD7.py                  # Support Vector Machines
├── TD8.py                  # Clustering
├── td9.py                  # Unsupervised Learning / PCA
├── requirements.txt        # Dépendances
├── .gitignore             # Fichiers à ignorer
└── README.md              # Ce fichier
```

## 🚀 Exécution Rapide

```bash
# Exécuter tous les TDs
for file in *.py; do echo "=== $file ===" && python "$file"; done

# Ou un par un
python TD1.py && python td2.py && python td3.py
```

## 📊 Concepts Clés Couverts

### Supervised Learning
- **Regression**: Prédire des valeurs continues
- **Classification**: Prédire des catégories

### Unsupervised Learning
- **Clustering**: Regrouper les données similaires
- **Dimensionality Reduction**: Réduire le nombre de features

### Model Evaluation
- **Train/Test Split**
- **Cross-Validation**
- **Métriques**: Accuracy, Precision, Recall, F1-Score
- **Confusion Matrix**

### Feature Engineering
- **Scaling et Normalization**
- **Encoding (One-hot, Label Encoding)**
- **Feature Selection**
- **Handling Imbalanced Data**

## 💡 Best Practices

✅ **À FAIRE**:
- Toujours normaliser vos features
- Utiliser la validation croisée
- Évaluer sur un test set indépendant
- Documenter vos expériences
- Comparer plusieurs modèles

❌ **À ÉVITER**:
- Data leakage (fit sur tout le dataset)
- Overfitting (modèle trop complexe)
- Ignorer l'imbalance des données
- Choisir un modèle sans justification

## 📚 Ressources Recommandées

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Pandas Documentation](https://pandas.pydata.org/)
- [Andrew Ng - Machine Learning Course](https://www.coursera.org/learn/machine-learning)
- [StatQuest with Josh Starmer](https://www.youtube.com/@statquest)
- [Hands-On Machine Learning Book](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/)

## 🔍 Conseils pour Déboguer

```python
# Vérifier la shape des données
print(X.shape, y.shape)

# Chercher les valeurs manquantes
print(df.isnull().sum())

# Vérifier la distribution des classes
print(y.value_counts())

# Tester d'abord sur un petit subset
X_sample = X[:100]
y_sample = y[:100]
```

## 📊 Exemple d'Utilisation Complète

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Charger les données
data = pd.read_csv('data.csv')

# 2. Préparer les features et la cible
X = data.drop('target', axis=1)
y = data['target']

# 3. Splitter
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 4. Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Entraîner
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

# 6. Évaluer
y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

## ❓ FAQ

**Q: Quelle est la différence entre TD et TP?**
- **TD**: Travaux Dirigés (exercices guidés)
- **TP**: Travaux Pratiques (implémentations)

**Q: Comment choisir entre plusieurs modèles?**
- Comparez les performances sur le test set
- Utilisez la validation croisée pour plus de robustesse
- Considérez la complexité et l'interpretabilité

## 📝 Licence

MIT License

## 👨‍💻 Auteur

**Younes Taibi**
- GitHub: [@taibi1995](https://github.com/taibi1995)

---

**Dernière mise à jour**: Février 2026
```

### 2️⃣ requirements.txt (À créer)

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
jupyter>=1.0.0
ipython>=7.0.0
scipy>=1.7.0
```

### 3️⃣ .gitignore (À créer)

```
# Jupyter Notebook
.ipynb_checkpoints/
*.ipynb_checkpoints

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
ENV/
env/
.venv/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Data files
data/
*.csv
*.xlsx
*.json

# Logs
*.log
```

---

## ✅ Actions à Effectuer sur GitHub

### 1. Ajouter une description
- Description: "Hands-on practical exercises in Machine Learning covering regression, classification, clustering, and feature engineering using Scikit-learn"

### 2. Ajouter les topics
- `machine-learning`
- `scikit-learn`
- `python`
- `education`
- `supervised-learning`
- `unsupervised-learning`
- `classification`
- `regression`

### 3. Pousser les fichiers
```bash
git add README.md requirements.txt .gitignore
git commit -m "docs: comprehensive ML documentation and dependencies"
git push origin main
```

---

## 📌 Suggestions d'Améliorations Futures
- Ajouter des datasets exemple pour chaque TD
- Créer des notebooks interactifs
- Ajouter des solutions partielles ou complètes
- Documenter les résultats attendus pour chaque TD
