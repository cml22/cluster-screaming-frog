import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Outil Maillage Interne - Analyse Multi-Étapes", layout="wide")
st.title("🔗 Outil de Maillage Interne - Analyse Étape par Étape")

# Étape 1 : Import du diagramme HTML
st.header("Étape 1 : Importer votre diagramme de clusters (HTML)")
uploaded_html = st.file_uploader("Importer le fichier HTML de diagramme de clusters (exporté depuis Screaming Frog)", type=["html"])

if uploaded_html:
    custom_html = uploaded_html.read().decode("utf-8")
    st.components.v1.html(custom_html, height=800, scrolling=True)
    
    st.success("Diagramme importé avec succès. Passez à l'étape suivante.")
    
    # Étape 2 : Import CSV
    st.header("Étape 2 : Importer vos données d'URL (CSV)")
    data = st.file_uploader("Importer un fichier CSV avec les URLs, mots-clés, etc.", type=['csv'])
    
    if data:
        df = pd.read_csv(data)
        st.write("Aperçu des données :", df.head())
        st.success("CSV chargé. Vous pouvez passer à l'étape suivante.")

        # Étape 3 : Analyse d'une URL ou d'un texte
        st.header("Étape 3 : Analysez une URL ou un contenu texte")
        input_method = st.radio("Souhaitez-vous analyser une URL ou du texte brut ?", ["URL", "Texte brut"])

        if input_method == "URL":
            url_input = st.text_input("Entrez l'URL à analyser")
            if url_input:
                try:
                    response = requests.get(url_input, timeout=10)
                    soup = BeautifulSoup(response.text, 'html.parser')
                    body_content = soup.body.get_text(separator=' ', strip=True)
                    st.success("Contenu récupéré avec succès depuis l'URL.")
                    input_text = body_content
                except Exception as e:
                    st.error(f"Erreur lors de la récupération de l'URL : {e}")
                    input_text = ""
        else:
            input_text = st.text_area("Collez votre texte à analyser ici")

        if input_text:
            # TF-IDF sur les contenus
            all_texts = df['keyword'].fillna('') + " " + df.get('content', pd.Series([''] * len(df)))
            corpus = all_texts.tolist() + [input_text]
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(corpus)
            cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()

            df['similarity'] = cosine_sim
            top_matches = df.sort_values(by='similarity', ascending=False).head(10)

            st.subheader("🔗 Suggestions de maillage sortant (vers où faire des liens depuis ce contenu)")
            st.dataframe(top_matches[['url', 'keyword', 'similarity']])

            st.subheader("🔗 Suggestions de maillage entrant (pages qui pourraient faire un lien vers ce contenu)")
            st.dataframe(top_matches[['url', 'keyword', 'similarity']])

            # Suggestions d'ancres
            st.subheader("🏷️ Suggestions d'ancres pour vos liens")
            anchors = []
            for kw in top_matches['keyword']:
                common_words = set(input_text.lower().split()) & set(kw.lower().split())
                anchors.append(", ".join(common_words) if common_words else "—")
            top_matches['suggested_anchor'] = anchors
            st.dataframe(top_matches[['url', 'suggested_anchor', 'similarity']])

    else:
        st.info("Veuillez importer votre fichier CSV à cette étape.")
else:
    st.info("Veuillez commencer par importer un fichier HTML du diagramme de clusters (Étape 1).")
