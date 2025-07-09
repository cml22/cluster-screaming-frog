import streamlit as st
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Outil de Maillage Interne", layout="wide")

st.title("🔗 Outil d'Optimisation du Maillage Interne")

# Import du diagramme HTML (obligatoire)
st.subheader("📂 Importer votre diagramme de clusters (HTML exporté depuis Screaming Frog)")
uploaded_html = st.file_uploader("Importer un fichier HTML de diagramme de clusters", type=["html"])

if uploaded_html:
    custom_html = uploaded_html.read().decode("utf-8")
    st.components.v1.html(custom_html, height=800, scrolling=True)
    
    with st.expander("ℹ️ Explication du diagramme de clusters"):
        st.markdown("""
        <div style="background-color: #f9f9f9; padding: 1.5em; border-radius: 10px; border: 1px solid #ccc; font-family: Arial, sans-serif;">
        <h3>📊 À propos du Diagramme de Clusters de Contenu</h3>
        <p>Ce diagramme représente une vue simplifiée du contenu de votre site web, basé sur les URL explorées. Chaque point correspond à une page, et les points proches entre eux partagent des similarités dans leur contenu.</p>
        
        <h4>Comment fonctionne ce diagramme ?</h4>
        <ul>
          <li><strong>Embeddings :</strong> Chaque page reçoit un vecteur numérique représentant sa sémantique.</li>
          <li><strong>Échantillonnage :</strong> Un sous-ensemble d'URL est affiché pour la lisibilité.</li>
          <li><strong>Réduction de dimensions :</strong> Les données sont compressées en 2D pour visualisation.</li>
          <li><strong>Clustering :</strong> Les pages sont regroupées en clusters colorés selon leur similarité.</li>
        </ul>
        <h4>📌 À propos des axes :</h4>
        <p>Les axes ne représentent pas une mesure précise. Ils servent à positionner les pages selon leur proximité sémantique.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Import CSV (URLs, clusters, coordonnées, mots-clés)
    st.subheader("📄 Importer vos données d'URL (CSV)")
    data = st.file_uploader("Importer un fichier CSV avec URLs, clusters, coordonnées et mots-clés", type=['csv'])
    
    if data:
        df = pd.read_csv(data)
        st.write("Aperçu des données :", df.head())

        # Recherche de mots-clés
        keyword = st.text_input("🔍 Rechercher un mot-clé")
        if keyword:
            results = df[df['keyword'].str.contains(keyword, case=False, na=False)]
            st.write(f"Résultats pour le mot-clé : {keyword}")
            st.dataframe(results)

        # Détection de cannibalisation
        st.subheader("🚨 Détection de cannibalisation")
        distance_threshold = st.slider("Seuil de proximité (plus bas = plus strict)", 0.0, 10.0, 1.0)
        if st.button("Analyser la cannibalisation"):
            coords = df[['x', 'y']].values
            dist_matrix = cdist(coords, coords)
            risks = []
            for i in range(len(df)):
                for j in range(i + 1, len(df)):
                    if dist_matrix[i, j] < distance_threshold and df.loc[i, 'cluster'] != df.loc[j, 'cluster']:
                        risks.append((df.loc[i, 'url'], df.loc[j, 'url'], dist_matrix[i, j]))
            if risks:
                st.write("Risques détectés :")
                st.table(risks)
            else:
                st.write("Aucun risque détecté.")

        # Recommandations de maillage
        st.subheader("🔗 Recommandations de Maillage")
        url_input = st.text_input("Entrez une URL pour obtenir des suggestions")
        if url_input:
            if url_input in df['url'].values:
                idx = df[df['url'] == url_input].index[0]
                target_coords = df.loc[idx, ['x', 'y']].values.reshape(1, -1)
                dist = cdist(target_coords, df[['x', 'y']].values).flatten()
                df['distance'] = dist
                nearby = df[(df['url'] != url_input)].sort_values('distance').head(10)

                st.write("👉 Pages vers lesquelles cette page pourrait faire des liens (sortants) :")
                st.dataframe(nearby[['url', 'keyword', 'distance']])

                st.write("👉 Pages qui pourraient faire un lien vers cette page (entrants) :")
                incoming = nearby.copy()
                st.dataframe(incoming[['url', 'keyword', 'distance']])
            else:
                st.error("L'URL saisie n'existe pas dans le fichier importé.")

        # Analyse sémantique optionnelle
        st.subheader("📄 Analyse sémantique (optionnelle)")
        if 'keyword' in df.columns:
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(df['keyword'].fillna(''))
            cosine_sim = cosine_similarity(tfidf_matrix)
            st.write("Similitude sémantique entre les pages (extrait 5x5) :")
            st.dataframe(pd.DataFrame(cosine_sim[:5, :5], index=df['url'].head(5), columns=df['url'].head(5)))
    else:
        st.info("Veuillez importer un fichier CSV pour analyser les données d'URL.")
else:
    st.info("💡 Veuillez importer un fichier HTML pour afficher votre diagramme de clusters.")
  
