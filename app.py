import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Outil de Maillage Interne - Analyse Multi-Étapes", layout="wide")
st.title("🔗 Outil de Maillage Interne - Analyse Multi-Étapes")

# Étape 1 : Import du diagramme HTML
st.header("Étape 1 : Importer votre diagramme de clusters (HTML)")
uploaded_html = st.file_uploader("Importer votre fichier HTML de diagramme (Screaming Frog ou autre)", type=["html"])

if uploaded_html:
    html_content = uploaded_html.read().decode("utf-8")
    
    # Note explicative sur le diagramme
    with st.expander("ℹ️ Explication du diagramme de clusters"):
        st.markdown("""
Le diagramme de clusters de contenu est une visualisation bidimensionnelle des URL de votre crawl, tracées et regroupées en clusters à partir des embeddings (vecteurs sémantiques) générés par IA.

**Les étapes clés :**
- **Génération d’embeddings** : Chaque page reçoit un vecteur IA représentant sa signification.
- **Échantillonnage** : Seul un sous-ensemble représentatif est affiché pour garder un diagramme lisible.
- **Réduction de dimensionnalité** : Les vecteurs (souvent >100 dimensions) sont réduits en 2D pour l'affichage.
- **Clustering** : Les points sont regroupés en clusters (couleurs différentes selon les groupes sémantiques).

Ce diagramme aide à détecter les thématiques proches, les silos, et les opportunités de maillage interne.
        """)

    st.components.v1.html(html_content, height=800, scrolling=True)

    search_url = st.text_input("🔎 Recherche d'URL dans le diagramme (centrage auto)")
    if search_url:
        st.markdown("""
Pour centrer automatiquement l'URL sur votre diagramme, votre fichier HTML doit contenir un identifiant ou un attribut associé aux URLs.
""")
        st.code("""
<script>
function centerNode(url) {
  const node = document.querySelector(`[data-url='${url}']`);
  if (node) {
    node.scrollIntoView({ behavior: 'smooth', block: 'center', inline: 'center' });
    node.style.border = '2px solid red';
  }
}
centerNode('%s');
</script>
""" % search_url, language='html')

# Étape 2 : Import CSV pour maillage interne
st.header("Étape 2 : Importer votre CSV d'URLs et contenus")
uploaded_csv = st.file_uploader("Importer un fichier CSV (colonnes : url, keyword, content)", type=["csv"])

if uploaded_csv:
    df = pd.read_csv(uploaded_csv)
    st.write("Aperçu des données :", df.head())

    # TF-IDF sur toutes les URLs du crawl
    corpus = df['keyword'].fillna('') + " " + df.get('content', pd.Series([''] * len(df)))
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    cosine_sim = cosine_similarity(tfidf_matrix)

    # Maillage sortant
    maillage = []
    for i, url in enumerate(df['url']):
        sim_scores = list(enumerate(cosine_sim[i]))
        sim_scores = [x for x in sim_scores if x[0] != i]
        sim_scores.sort(key=lambda x: x[1], reverse=True)
        top_outgoing = sim_scores[:5]
        for idx, score in top_outgoing:
            kw1 = df.at[i, 'keyword'].lower().split()
            kw2 = df.at[idx, 'keyword'].lower().split()
            anchor = ", ".join(set(kw1) & set(kw2)) or "—"
            maillage.append({
                'from_url': url,
                'to_url': df.at[idx, 'url'],
                'similarity': score,
                'anchor': anchor
            })
    maillage_df = pd.DataFrame(maillage)
    st.subheader("🔗 Maillage interne automatique (liens sortants)")
    st.dataframe(maillage_df)

# Étape 3 : Analyse en Bulk
st.header("Étape 3 : Analyse Bulk d'URLs ou de contenus")
bulk_option = st.radio("Méthode d'import :", ["Importer un CSV", "Entrer manuellement"])

if bulk_option == "Importer un CSV":
    bulk_file = st.file_uploader("Importer votre CSV (colonnes : url, content)", type=["csv"], key="bulk_csv")
    if bulk_file:
        bulk_df = pd.read_csv(bulk_file)
        st.write("Contenus importés :", bulk_df.head())
elif bulk_option == "Entrer manuellement":
    urls_input = st.text_area("Collez ici vos URLs (1 par ligne)")
    contents_input = st.text_area("Collez ici vos contenus (1 par ligne, dans le même ordre que les URLs)")
    if urls_input and contents_input:
        urls = urls_input.strip().split('\n')
        contents = contents_input.strip().split('\n')
        bulk_df = pd.DataFrame({'url': urls, 'content': contents})

if 'bulk_df' in locals():
    bulk_corpus = bulk_df['content'].fillna('') + " " + bulk_df.get('url', pd.Series([''] * len(bulk_df)))
    combined_corpus = pd.concat([pd.Series(corpus), pd.Series(bulk_corpus)], ignore_index=True)
    tfidf_matrix = vectorizer.fit_transform(combined_corpus)
    
    # Analyse entre bulk et diagramme
    bulk_results = []
    for i in range(len(df), len(combined_corpus)):
        sim_scores = cosine_similarity(tfidf_matrix[i], tfidf_matrix[:len(df)]).flatten()
        top_matches = sim_scores.argsort()[::-1][:5]
        for idx in top_matches:
            kw1 = bulk_corpus.iloc[i - len(df)].lower().split()
            kw2 = corpus.iloc[idx].lower().split()
            anchor = ", ".join(set(kw1) & set(kw2)) or "—"
            bulk_results.append({
                'source_bulk_url': bulk_df.iloc[i - len(df)]['url'],
                'target_site_url': df.iloc[idx]['url'],
                'similarity': sim_scores[idx],
                'anchor': anchor
            })
    bulk_df_outgoing = pd.DataFrame(bulk_results)
    st.subheader("🔗 Suggestions de maillage sortant (depuis votre bulk vers votre site)")
    st.dataframe(bulk_df_outgoing)

    # Analyse liens entrants (site → bulk)
    bulk_results_in = []
    for i in range(len(df)):
        sim_scores = cosine_similarity(tfidf_matrix[i], tfidf_matrix[len(df):]).flatten()
        top_matches = sim_scores.argsort()[::-1][:5]
        for idx in top_matches:
            kw1 = corpus.iloc[i].lower().split()
            kw2 = bulk_corpus.iloc[idx].lower().split()
            anchor = ", ".join(set(kw1) & set(kw2)) or "—"
            bulk_results_in.append({
                'site_url': df.iloc[i]['url'],
                'bulk_target_url': bulk_df.iloc[idx]['url'],
                'similarity': sim_scores[idx],
                'anchor': anchor
            })
    bulk_df_incoming = pd.DataFrame(bulk_results_in)
    st.subheader("🔗 Suggestions de maillage entrant (depuis votre site vers votre bulk)")
    st.dataframe(bulk_df_incoming)
