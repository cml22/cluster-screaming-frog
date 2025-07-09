import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

st.set_page_config(page_title="Outil de Maillage Interne - Analyse Automatique", layout="wide")
st.title("🔗 Outil de Maillage Interne - Analyse Automatique")

# Étape 1 : Import du diagramme HTML
st.header("Étape 1 : Importer votre diagramme de clusters (HTML)")
uploaded_html = st.file_uploader("Importer le fichier HTML de votre diagramme (exporté depuis Screaming Frog)", type=["html"])

if uploaded_html:
    html_content = uploaded_html.read().decode("utf-8")
    st.components.v1.html(html_content, height=800, scrolling=True)

    with st.expander("ℹ️ Explication du diagramme de clusters"):
        st.markdown("""
Le diagramme de clusters de contenu est une représentation 2D des pages de votre site, regroupées selon leur proximité sémantique :
- **Génération d’embeddings** : chaque URL reçoit un vecteur de signification.
- **Échantillonnage** : un sous-ensemble est sélectionné pour la clarté visuelle.
- **Réduction de dimensions** : pour afficher les points sur 2 axes.
- **Clustering** : les groupes sont affichés en différentes couleurs pour identifier les silos thématiques.
""")

    # Extraction des URLs présentes dans le fichier HTML (par regex, plus robuste)
    urls = list(set(re.findall(r'https?://[^\s"\'<>]+', html_content)))
    st.success(f"{len(urls)} URLs extraites du diagramme.")

    if len(urls) == 0:
        st.warning("Aucune URL détectée dans votre fichier HTML. Vérifiez sa structure ou contactez votre équipe SEO.")
    else:
        # Étape 2 : Analyse
        st.header("Étape 2 : Analyse d'une URL ou d'un contenu texte")
        method = st.radio("Choisissez votre méthode :", ["URL", "Texte brut"])

        input_text = ""
        run_analysis = False

        if method == "URL":
            url_input = st.text_input("Entrez une URL")
            if url_input:
                try:
                    response = requests.get(url_input, timeout=10)
                    page_soup = BeautifulSoup(response.text, 'html.parser')
                    input_text = page_soup.get_text(separator=' ', strip=True)
                    st.success("Contenu récupéré avec succès.")
                    run_analysis = True
                except Exception as e:
                    st.error(f"Erreur lors de la récupération de l'URL : {e}")
        else:
            input_text = st.text_area("Collez votre contenu ici")
            if st.button("Analyser le contenu"):
                run_analysis = True

        if run_analysis and input_text.strip():
            # Téléchargement du contenu des URLs du diagramme
            diagram_texts = []
            for u in urls:
                try:
                    r = requests.get(u, timeout=5)
                    s = BeautifulSoup(r.text, 'html.parser')
                    text = s.get_text(separator=' ', strip=True)
                    diagram_texts.append(text)
                except Exception:
                    diagram_texts.append("")

            # Analyse TF-IDF
            corpus = diagram_texts + [input_text]
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(corpus)

            # Similarité avec le contenu fourni
            similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
            df = pd.DataFrame({'url': urls, 'similarity': similarities})

            # Maillage sortant (de mon contenu vers les autres)
            outgoing_links = df.sort_values(by='similarity', ascending=False).head(10)
            outgoing_links['anchor'] = outgoing_links['url'].apply(
                lambda u: " ".join(set(input_text.lower().split()) & set(u.lower().split())) or "—"
            )

            st.subheader("🔗 Maillage sortant suggéré (vers des pages du diagramme)")
            st.dataframe(outgoing_links[['url', 'similarity', 'anchor']])

            # Maillage entrant (pages qui pourraient faire un lien vers mon contenu)
            incoming_links = outgoing_links.copy()
            st.subheader("🔗 Maillage entrant suggéré (depuis des pages du diagramme vers votre contenu)")
            st.dataframe(incoming_links[['url', 'similarity', 'anchor']])
else:
    st.info("Veuillez d'abord importer votre fichier HTML du diagramme.")
