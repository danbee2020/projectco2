from PIL import Image
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Configuration de la page
st.set_page_config(
    page_title="Rapport d'Analyse",
    page_icon="📊",
    layout="wide")


# fonction pour Charger le dataframe

df = pd.read_csv("data-2.csv")



# Sidebar pour la navigation
st.sidebar.title("📑 Table des matières")
page = st.sidebar.radio("",

    ["Présentation du projet", "Exploration des Données", "Visualisations", "Modélisation", "Conclusions"])


# Page Présentation du projet
if page == "Présentation du projet":
    st.title("📋Emission de CO2 par les véhicules")
   

    st.image("pic-co2.jpg",caption="Mon image", width=600)
    
    st.header("Contexte")
    st.write("""L'émission de CO2 est une variable constamment mesuré car elle ne contribue pas à préserver notre planète. """)        
    st.write("""Les émissions de CO2 identifie les véhicules qui émettent le plus de CO2 pour identifier les caractéristiques techniques qui jouent un rôle dans la pollution. """)
    st.write("""Comme prévenir c'est guérir prédire à l’avance cette pollution permet de prévenir dans le cas de l’apparition de nouveaux types de véhicules (nouvelles séries de voitures par exemple)""")

  
    
    st.header("Méthodologie")
    st.write("""l'analyse consistera Chargement et exploration exhaustif des données.Analyse statistique et visualisations.Machine learning.Prédiction du CO2. Interprétabilité du modèle (SHAP) """)




# Page Exploration des Données


elif page == "Exploration des Données":
    st.title("🔍 Exploration des Données")
    
    st.header("Vue d'ensemble du dataset")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dimensions du dataset")
        st.write(f"Nombre de lignes : {df.shape[0]}")
        st.write(f"Nombre de colonnes : {df.shape[1]}")
    
    with col2:
        st.subheader("Types des variables")
        st.write(df.dtypes)
    
        st.header("Statistiques descriptives")
        st.write(df.describe())
    
        st.header("Analyse des valeurs manquantes")
        missing_values = df.isnull().sum()
        st.write(missing_values[missing_values > 0])

# Page Visualisation
elif page == "Visualisations":
     st.title("📈 Visualisations")
    
     st.header("Distribution des variables")
    # Exemple de visualisation avec Plotly
     col = st.selectbox("Sélectionnez une variable", df.select_dtypes(include=[np.number]).columns)
     fig = px.histogram(df, x=col, title=f"Distribution de {col}")
     st.plotly_chart(fig)
    
     st.header("Analyse des corrélations")
    # Matrice de corrélation
     corr = df.select_dtypes(include=[np.number]).corr()
     fig = px.imshow(corr, 
                    title="Matrice de corrélation",
                    color_continuous_scale='RdBu')
     st.plotly_chart(fig)
    
     st.header("Analyses bivariées")
    # Scatter plot
     x_col = st.selectbox("Variable X", df.select_dtypes(include=[np.number]).columns)
     y_col = st.selectbox("Variable Y", df.select_dtypes(include=[np.number]).columns)
     fig = px.scatter(df, x=x_col, y=y_col, title=f"{x_col} vs {y_col}")
     st.plotly_chart(fig)

# Page Modélisation
elif page == "Modélisation":
     st.title("🤖 Modélisation")
    
     st.header("Préparation des données")
    # Exemple avec régression linéaire
     target = st.selectbox("Variable cible", df.select_dtypes(include=[np.number]).columns)
     features = st.multiselect("Variables explicatives",
                            [col for col in df.select_dtypes(include=[np.number]).columns 
                             if col != target])
    
     if features:
        # Préparation des données
        X = df[features]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Standardisation
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Modélisation
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        # Résultats
        st.header("Résultats du modèle")
        st.write(f"R² Score: {r2_score(y_test, y_pred):.3f}")
        st.write(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.3f}")
        
        # Coefficients
        coef_df = pd.DataFrame({
            'Variable': features,
            'Coefficient': model.coef_
        })
        st.write("Coefficients du modèle:")
        st.write(coef_df)

# Page Conclusions
else:
    st.title("📝 Conclusions et Recommandations")
    
    st.header("Synthèse des résultats")
    st.write("""
    [Résumez ici les principales découvertes de votre analyse]
    """)
    
    st.header("Recommandations")
    st.write("""
    [Listez vos recommandations basées sur l'analyse]
    """)
    
    st.header("Limites et perspectives")
    st.write("""
    [Discutez des limites de l'analyse et des pistes d'amélioration]
    """)

# Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("Créé par [Votre Nom]")