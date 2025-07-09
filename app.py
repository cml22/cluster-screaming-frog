import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Outil Maillage Interne - Analyse Multi-Étapes", layout="wide")
st.title("🔗 Outil de Maillage Interne - Analyse Multi-Étapes")

# Étape 1 : Import diagramme HTML
st.header("Étape 1 : Importer votre diagramme de clusters (HTML)")
uploaded_html = st.file_uploader("Importer le fichier HTML du diagramme (depuis Screaming Frog ou autre)", type=["html"])

if uploaded_html:
    html_content = uploaded_html.read().decode("utf-8")
    st.components.v1.html(html_content, height=800, scrolling=True)
    st.success("Diagramme chargé.")

    search_url = st.text_input("🔎 Recherche d'URL dans le diagramme (nécessite script JS dans le HTML)")
    if search_url:
        st.info(f"Recherche pour : {search_url} (pour le surlignage, ajouter un script JS dans le diagramme HTML)")

    # Étape 2 : Import CSV
    st.header("Étape 2 : Importer votre CSV d'URLs & contenus")
    uploaded_csv = st.file_uploader("Importer un fichier CSV (colonnes : url, keyword, content)", type=["csv"])

    if uploaded_csv:
        df = pd.read_csv(uploaded_csv)
        st.write("Aperçu des données :", df.head())
        
        # TF-IDF
        contents = df['keyword'].fillna('') + " " + df.get('content', pd.Series([''] * len(df)))
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(contents)
        cosine_sim = cosine_similarity(tfidf_matrix)

        # Génération du maillage automatique : Sortants + Entrants
        maillage = []
        for i, url in enumerate(df['url']):
            sim_scores = list(enumerate(cosine_sim[i]))
            sim_scores = [x for x in sim_scores if x[0] != i]
            sim_scores.sort(key=lambda x: x[1], reverse=True)
            top_outgoing = sim_scores[:5]  # Top 5 liens sortants
            
            for idx, score in top_outgoing:
                kw1 = df.at[i, 'keyword'].lower().split()
                kw2 = df.at[idx, 'keyword'].lower().split()
                common_words = set(kw1) & set(kw2)
                anchor = ", ".join(common_words) if common_words else "—"
                maillage.append({
                    'from_url': url,
                    'to_url': df.at[idx, 'url'],
                    'similarity': score,
                    'anchor': anchor
                })

        maillage_df = pd.DataFrame(maillage)
        st.subheader("🔗 Maillage interne automatique (liens sortants)")
        st.dataframe(maillage_df)

        # Génération des liens entrants (inverse)
        st.subheader("🔗 Suggestions de liens entrants")
        reverse_links = []
        for target_url in df['url']:
            incoming = maillage_df[maillage_df['to_url'] == target_url]
            incoming_sorted = incoming.sort_values(by='similarity', ascending=False).head(5)
            for _, row in incoming_sorted.iterrows():
                reverse_links.append({
                    'to_url': target_url,
                    'from_url': row['from_url'],
                    'similarity': row['similarity'],
                    'anchor': row['anchor']
                })
        
        reverse_df = pd.DataFrame(reverse_links)
        st.dataframe(reverse_df)

        # Étape 3 : Analyse ciblée (comme avant)
        st.header("🧐 Analyse ciblée d'une URL ou d'un contenu texte")
        choice = st.radio("Analyser une URL ou un texte brut ?", ["URL", "Texte brut"])

        if choice == "URL":
            input_url = st.text_input("Entrez l'URL à analyser")
            if input_url:
                try:
                    response = requests.get(input_url, timeout=10)
                    soup = BeautifulSoup(response.text, 'html.parser')
                    input_text = soup.body.get_text(separator=' ', strip=True)
                    st.success("Contenu récupéré avec succès.")
                except Exception as e:
                    st.error(f"Erreur : {e}")
                    input_text = ""
        else:
            input_text = st.text_area("Collez votre texte ici")

        if input_text:
            corpus = contents.tolist() + [input_text]
            tfidf_matrix = vectorizer.fit_transform(corpus)
            similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
            df['similarity'] = similarity_scores
            top_matches = df.sort_values(by='similarity', ascending=False).head(10)
            
            st.subheader("🔗 Suggestions de maillage sortant (pour ce contenu)")
            st.dataframe(top_matches[['url', 'keyword', 'similarity']])

            st.subheader("🏷️ Suggestions d'ancres")
            anchors = []
            for kw in top_matches['keyword']:
                common = set(input_text.lower().split()) & set(kw.lower().split())
                anchors.append(", ".join(common) if common else "—")
            top_matches['suggested_anchor'] = anchors
            st.dataframe(top_matches[['url', 'suggested_anchor', 'similarity']])
    else:
        st.info("Veuillez importer votre CSV.")
else:
    st.info("Veuillez d'abord importer un diagramme HTML.")
