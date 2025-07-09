import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Recommandations de Maillage Automatique", layout="wide")
st.title("🔗 Outil de Recommandation Automatique de Maillage Interne")

# Import CSV
st.subheader("📄 Importer vos données d'URL (CSV)")
data = st.file_uploader("Importer un fichier CSV avec les URLs, mots-clés, etc.", type=['csv'])

if data:
    df = pd.read_csv(data)
    st.write("Aperçu des données :", df.head())

    # Input : URL ou texte brut
    st.subheader("💬 Analyse d'un contenu")
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
        all_texts = df['keyword'].fillna('') + " " + df.get('content', pd.Series([''] * len(df)))  # Si tu veux ajouter les contenus plus tard
        corpus = all_texts.tolist() + [input_text]  # Texte utilisateur en dernier
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(corpus)
        cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()

        df['similarity'] = cosine_sim
        top_matches = df.sort_values(by='similarity', ascending=False).head(10)

        st.subheader("🔗 Pages vers lesquelles vous pourriez faire un lien (liens sortants)")
        st.dataframe(top_matches[['url', 'keyword', 'similarity']])

        st.subheader("🔗 Pages qui pourraient faire un lien vers votre contenu (liens entrants)")
        st.dataframe(top_matches[['url', 'keyword', 'similarity']])

        # Suggestions d'ancres (mots-clés communs)
        st.subheader("🏷️ Suggestions d'ancres pour vos liens")
        anchors = []
        for kw in top_matches['keyword']:
            common_words = set(input_text.lower().split()) & set(kw.lower().split())
            anchors.append(", ".join(common_words) if common_words else "—")
        top_matches['suggested_anchor'] = anchors
        st.dataframe(top_matches[['url', 'suggested_anchor', 'similarity']])
else:
    st.info("Veuillez importer un fichier CSV avant d'utiliser l'outil.")
