import pandas as pd
import streamlit as st
import joblib
from datetime import datetime

# Charger les modèles XGBoost
model_path_regression = r'C:\Users\Islem\Desktop\prediction_delais\xgboost_model.pkl'
model_path_classification = r'C:\Users\Islem\Desktop\prediction_delais\xgboostmodel.pkl'
model_regression = joblib.load(model_path_regression)
model_classification = joblib.load(model_path_classification)

# Fonction pour préparer les données d'entrée
def prepare_input_data(montant, date_emission, type_mandat, type_identite_expediteur, notes, bureau_emiss):
    # Conversion de la date d'émission en datetime
    date_emission = pd.to_datetime(date_emission)

    # Création d'un DataFrame avec les colonnes attendues
    input_data = pd.DataFrame({
        "Mand_Montant": [montant],
        "Mand_DateEmiss": [date_emission],
        "CodeTypeMandat_MCRBT": [1 if type_mandat == 'MCRBT' else 0],
        "CodeTypeMandat_MEXP": [1 if type_mandat == 'MEXP' else 0],
        "CodeTypeMandat_MSERV": [1 if type_mandat == 'MSERV' else 0],
        "Mand_TypeIdentiteExp_P": [1 if type_identite_expediteur == 'Passeport' else 0],
        "Mand_TypeIdentiteExp_S": [1 if type_identite_expediteur == 'Carte de Séjour' else 0],
        "Mand_Observation_RP": [1 if notes == 'RP' else 0],
        "Mand_TypeIdentite_P": [1 if type_identite_expediteur == 'Passeport' else 0],  # Ajustez selon vos besoins
        "Mand_TypeIdentite_S": [1 if type_identite_expediteur == 'Carte de Séjour' else 0]  # Ajustez selon vos besoins
    })

    # Extraction de l'année, mois et jour de la date d'émission
    input_data["Mand_YearEmiss"] = input_data["Mand_DateEmiss"].dt.year
    input_data["Mand_MonthEmiss"] = input_data["Mand_DateEmiss"].dt.month
    input_data["Mand_DayEmiss"] = input_data["Mand_DateEmiss"].dt.day

    # Pour simplifier, on assigne les mêmes valeurs aux colonnes de la date de paiement (Mand_YearPay, etc.)
    input_data["Mand_YearPay"] = input_data["Mand_YearEmiss"]
    input_data["Mand_MonthPay"] = input_data["Mand_MonthEmiss"]
    input_data["Mand_DayPay"] = input_data["Mand_DayEmiss"]

    # Ajout d'une colonne pour Mand_BurEmiss (bureau émetteur saisi par l'utilisateur)
    input_data["Mand_BurEmiss"] = bureau_emiss

    # Vérification des colonnes manquantes et ajout de valeurs par défaut
    columns_required = [
        'Mand_BurEmiss', 'Mand_Montant', 'Mand_YearEmiss', 'Mand_MonthEmiss', 'Mand_DayEmiss',
        'Mand_YearPay', 'Mand_MonthPay', 'Mand_DayPay', 'CodeTypeMandat_MCRBT', 'CodeTypeMandat_MEXP',
        'CodeTypeMandat_MSERV', 'Mand_TypeIdentiteExp_P', 'Mand_TypeIdentiteExp_S', 'Mand_Observation_RP',
        'Mand_TypeIdentite_P', 'Mand_TypeIdentite_S'
    ]

    # Sélectionner uniquement les colonnes attendues par le modèle
    input_data = input_data[columns_required]

    return input_data

# Configuration de l'interface Streamlit (couleurs personnalisées)
st.set_page_config(
    page_title="Prédiction des Délais de Paiement", 
    page_icon="🖼️", 
    layout="wide",  # Pour utiliser toute la largeur de la page
    initial_sidebar_state="collapsed"
)

# Personnaliser le thème : Jaune et Bleu comme la Poste Tunisienne
st.markdown("""
    <style>
        body {
            background-color: #F6D02F; /* Jaune Poste Tunisienne */
        }
        .css-1d391kg { 
            background-color: #003D73; /* Bleu Poste Tunisienne */
        }
        .css-1aeh3xv { 
            color: #003D73;
        }
        .stButton>button {
            background-color: #003D73;
            color: white;
            border-radius: 10px;
        }
        .stSelectbox>div>div>input {
            background-color: #003D73;
            color: white;
        }
        .stTextInput>div>div>input {
            background-color: #003D73;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# Layout Divisé en 2 colonnes (pour le logo et titre à gauche, les autres éléments à droite)
col1, col2 = st.columns([1, 3])

# Afficher le logo dans la première colonne
with col1:
    st.image(r'C:\Users\Islem\Desktop\prediction_delais\logo poste tunisienne.png', width=150)

# Afficher le titre dans la deuxième colonne
with col2:
    st.title("Prédiction des Délais de Paiement")

# Séparer la partie saisie des données et la partie résultats
st.header("Veuillez entrer les informations suivantes")

# Formulaire de saisie des informations
col3, col4 = st.columns(2)

with col3:
    montant = st.number_input("Montant du mandat", value=100.0)
    date_emission = st.date_input("Date d'émission du mandat", value=datetime.now())
    type_mandat = st.selectbox("Type de mandat", ['M1406', 'MCRBT', 'MEXP', 'MSERV'])

with col4:
    type_identite_expediteur = st.selectbox("Type d'identité de l'expéditeur", ['CIN', 'Passeport', 'Carte de Séjour'])
    notes = st.text_input("Observations", value="")
    bureau_emiss = st.number_input("Bureau émetteur", value=1)

# Préparer les données pour la prédiction
prepared_data = prepare_input_data(montant, date_emission, type_mandat, type_identite_expediteur, notes, bureau_emiss)

# Bouton de prédiction
if st.button("Prédire"):
    # Prédiction de la durée exacte (régression)
    prediction_regression = model_regression.predict(prepared_data)
    # Prédiction de la classe de durée (classification)
    prediction_classification = model_classification.predict(prepared_data)
    
    # Affichage des résultats dans un cadre central en bas
    st.markdown("<hr>", unsafe_allow_html=True)  # Séparation visuelle

    with st.container():
        st.subheader("Résultats de la Prédiction")
        st.write(f"**Durée exacte prédite :** {prediction_regression[0]} jours")
        st.write(f"**Classe de durée prédite :** {prediction_classification[0]}")
