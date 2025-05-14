####Importation des bibliothèques
 
import streamlit as st
import pandas as pd
import joblib

####Configuration de la page

####Titre principal

st.title ("🎬 Estimation du revenu d’un film (en millions $)")

####Chargement du modèle

try: #On va anticiper une erreur qui pourrait se produire dans le bloc.
    model = joblib.load("movies_success_predict.pkl")
    #Essaie de charger le modèle enregistré
    #qui contient notre modèle de prédiction de succès de films.
    #est une fonction qui permet de lire) le fichier de joblib
    
    expected_cols = joblib.load("new_columns.pkl") 
    #Essaie de charger un autre fichier enregistré 
    #qui contient la liste des informations (colonnes) importantes de notre modèle.
    #la variable va s'assurer que les données donner au modèle pour faire la prédiction sont
    #les bonnes informations et dans le bon ordre.
    
    st.success ("✅ Modèle chargé avec succès.") #fonction de streamlit
    #Si les 2 chargements ont réussi il affiche un message de succès.

except Exception as e: #définit ce qui doit se passer si une erreur se produit une exception,on utilise pr python.
    st.error (f"❌ Erreur lors du chargement du modèle : {e}") 
    #f-string permet d'insérer 
    #des valeurs de variables dans une chaîne de caractères
    #Si une erreur se produit pendant le chargement d'un des fichiers 
    #il affiche un msg d'erreur.
    #{e} insère la valeur de la variable dans la chaîne de caractères.

####Liste fixe des genres 
genres = ['Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime',
          'Drama', 'Family', 'Fantasy', 'History', 'Horror', 'Music',
          'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Sport', 'Thriller',
          'War', 'Western']

#on assigne une liste de chaînes de caractères à la variable 'genres'
#qui contient tous les genres de films possibles que notre app va utiliser. 

####Extraction des choix depuis le CSV

df_raw = pd.read_csv("movies_success_predict.csv")
#on crée une variable df_raw. Elle va contenir des informations qui
#pourrait contenir des colonnes telles que le titre du film, les acteurs ect avec d'autres caractéristiques. 

directors = sorted(df_raw["Director"].dropna().unique().tolist())
#on va chercher dans le df df_raw la colonne "Director". 
#la colonne contient les noms des réalisateurs de chqu film.
#.dropna() : Elle va supprimer tt les valeurs manquantes de la colonne.
#.unique() : Après qu'on supprime les valeurs manquantes, la fonction va extraire tt les valeurs uniques
#.tolist() : on convertit cette liste unique avec les noms des réalisateurs 
#la fonction sorted() prend la liste et trie par ordre alphabétique.

actors = sorted(df_raw["Actors"].dropna().unique().tolist())
#On fait la même chose que pour les réalisateurs

####utilisateur

genre = st.multiselect("🎞️ Genres", genres, default=['Action', 'Adventure'])
#une fonction de Streamlit. 
#Elle crée une interface utilisateur interactive où l'utilisateur peut sélectionner + options parmi la liste.
#c une liste déroulante où on peux cocher plusieurs cases.
#Défaut va sélectionnées par défaut lorsque l'app se charge pour la 1e fois

director = st.selectbox("🎬 Réalisateur", directors) #une autre fonction de Streamlit. 
#Elle crée une liste déroulante à partir des données de directors.

actors_choice = st.multiselect("👥 Acteurs principaux", actors, default=actors[:1])
#on prend le premier élément de la liste actors et la sélectionne par défaut

year = st.number_input("📅 Année de sortie", min_value=1900, max_value=2100, value=2024)
#une fonction Streamlit qui crée un champ qui permet à l'utilisateur d'entrer un nbr
#on lui donne une valeur mini et max pour l'année de sortie du film.

runtime = st.slider("⏱️ Durée (minutes)", min_value=30, max_value=300, value=120)
#C'est une fonction Streamlit qui crée un curseur horizontal 
#pour que l'utilisateur peut déplacer pour sélectionner une valeur.

rating = st.slider("⭐ Note IMDb", min_value=0.0, max_value=10.0, step=0.1, value=7.0) 
#On crée une variable rating pour stocker la note IMDb sélectionnée.

metascore = st.slider("🧠 Metascore", min_value=0, max_value=100, value=70)
#On crée une variable metascore pour stocker le Metascore sélectionné et toujours un curseur Streamlit.

#bref toutes les lignes de code utilisent la librairie Streamlit pour créer des éléments interactifs sur la page web.

####Préparation des données

genre_str = ",".join(genre)
#On crée une variable genre_str qui va contenir une chaîne de caractères.
#join() est une fonction qui s'applique à une chaîne de caractères. Elle prend en argument la liste et elle va joindre tt les éléments de cette liste 
#en une seule chaîne de caractères en utilisant la virgule comme séparateur entre chaque élément.

actors_str = ", ".join(actors_choice)
#On crée une variable actors_str qui va contenir une chaîne de caractères.
#Tous les noms d'acteurs de cette liste seront combinés en une seule chaîne.

input_data = { 
    "Director": director,
    "Actors": actors_str,
    "Year": year,
    "Runtime (Minutes)": runtime,
    "Rating": rating,
    "Metascore": metascore
}
#on crée une variable nommée input_data. 
#Cette variable va contenir un dictionnaire qui contient toutes les donées que l'utilisateur a entrées
#Chaque information est stockée comme une valeur associé càd que le dictionnaire est utilisé pour alimenter 
#le modèle de prédiction afin de déterminer le succès potentiel du film basé sur ces entrées.

####Conversion en DataFrame

df_input = pd.DataFrame([input_data]) 
#une variable appelée df_input. Elle contient un DF de pandas.Chaque clé du dictionnaire devient le nom d'une colonne
#les valeurs du dictionnaire remplissent cette unique ligne.
#C une étape nécessaire car notre modèle de machine learning fonctionnent mieux avec des données du DF.

df_input = pd.get_dummies(df_input)
#On réutilise la variable df_input. elle contiendra une version modifiée du DF.
#Cela prépare les données d'entrée dans un format adapté pour être utilisé par un modèle de ML pour faire une prédiction

####Compléter avec colonnes manquantes

for col in expected_cols: 
    if col not in df_input.columns:
        df_input[col] = 0
df_input = df_input[expected_cols]
#Elle va parcourir chaque élément de la liste expected_cols. À chaque itération de la boucle, 
#l'élément de la liste sera stocké dans la variable col. À l'intérieur de la boucle, 
#cette condition vérifie si le nom de colonne actuel (col) existe déjà dans les colonnes de notre DF 
# df_input. df_input.columns donne une liste de tt les noms des colonnes du DF 
#L'opérateur not in vérifie si col n'est pas présent dans cette liste.
#Si la condition if est vraie càd si la colonne attendue n'existe pas dans df_input, 
#cette ligne ajoute une nouvelle colonne au DF df_input avec le nom de la colonne manquante (col) et 
#remplit tt les cellules de cette nouvelle colonne avec la valeur 0.

####Affichage des données utilisateur

st.subheader("Données saisies par l'utilisateur") 
#On crée un sous-titre pour la section des données saisies par l'utilisateur.
#st.subheader est une fonction de Streamlit 

st.dataframe(df_input)
#On affiche le DF df_input dans l'app Streamlit.

####Prédiction

if st.button("Prédire le revenu estimé"):
    try:
        prediction = model.predict(df_input)[0]
        st.success(f"💰 Revenu estimé : **{prediction:.2f} millions de dollars**")
    except Exception as e:
        st.error(f"❌ Erreur lors de la prédiction : {e}")
#On crée un bouton "Prédire le revenu estimé".
#Lorsque l'utilisateur clique sur ce bouton, le modèle de prédiction est exécuté et le résultat est affiché.
#La fonction predict() du modèle est utilisée pour faire la prédiction. Le résultat est affiché avec un message de succès.
#La valeur de la prédiction est formatée pour afficher 2 décimales.
#Si une erreur se produit pendant la prédiction, un message d'erreur est affiché.