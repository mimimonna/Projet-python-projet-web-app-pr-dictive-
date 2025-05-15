🎬 Prédiction du Revenu d’un Film

Bienvenue dans notre projet python ! 
Ici, on prédit le revenu estimé (en millions de dollars) d’un film à partir de ses caractéristiques principales comme le genre, le réalisateur, les acteurs, la note IMDb, le metascore.
Pour réaliser ce travail, nous nous sommes appuyés sur un jeu de données complet et fiable provenant du site Kaggle : https://www.kaggle.com/datasets/therealsampat/predict-movie-success-rate.


📌 L'Objectif

Ce projet a pour but de développer un modèle de machine learning capable de prédire le revenu d’un film sans se baser uniquement sur les chiffres passés mais plutôt sur des éléments connus avant sa sortie. L’application permet de tester différents scénarios facilement.


🛠️ Technologies utilisées

- Python
- Pandas & Scikit-learn
- RandomForestRegressor
- Streamlit
- Joblib


🧠 Description du modèle

Nous avons utilisé un modèle Random Forest Regressor pour entraîner les données.
Les variables utilisées pour la prédiction sont :

- 🎞️ Genre
- 🎬 Réalisateur
- 👥 Acteurs
- 📅 Année de sortie
- ⏱️ Durée (minutes)
- ⭐ Note IMDb
- 🧠 Metascore.
  
Toutes les variables catégorielles (genre, réalisateur, etc.) sont encodées avec get_dummies.


💻 Lancement de l'application

Cloner le repositoire, il faut installer les dépendances :
pip install -r requirements.txt

Lancer l'application Streamlit :
streamlit run movies_success_predict.py


📂 Organisation des fichiers


├── data/

   └── movies_success_predict.csv

├── model/

   └── movies_success_predict.pkl

   └── new_columns.pkl

├── movies_success_predict.py

├── movies_success_predict.ipynb

├── README.md


Ce projet nous a permis de mettre en pratique nos compétences en machine learning et en développement d’interface utilisateur.

Ce projet à été crée par Lilliane, Hugo et Monna , nous sommes des étudiants en Data Analytic et Intelligence Artificiel à L'université Panthéon Sorbonne.
