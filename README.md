# 3️⃣ PROJET: TPs-Machine-learning

## 📋 Informations Actuelles
- **Lien**: https://github.com/taibi1995/TPs-Machine-learning
- **Type**: Travaux Pratiques / Education
- **Langage**: Python (100%)
- **Fichiers**: 9 fichiers (TD1-TD9)


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

