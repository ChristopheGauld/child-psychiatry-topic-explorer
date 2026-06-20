from __future__ import annotations

import re
import unicodedata
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

from config import R_EXTRA_STOPWORDS, SEED

FRENCH_STOPWORDS = {
    "alors", "au", "aucun", "aussi", "autre", "aux", "avec", "avoir", "bon", "car",
    "ce", "cela", "ces", "ceux", "chaque", "ci", "comme", "comment", "dans", "de", "des",
    "du", "dedans", "dehors", "depuis", "devrait", "doit", "donc", "dos", "début", "elle",
    "elles", "en", "encore", "essai", "est", "et", "eu", "fait", "faites", "fois", "font",
    "hors", "ici", "il", "ils", "je", "juste", "la", "le", "les", "leur", "là", "ma",
    "maintenant", "mais", "mes", "mine", "moins", "mon", "mot", "même", "ni", "nommés",
    "notre", "nous", "ou", "où", "par", "parce", "pas", "peut", "peu", "plupart", "pour",
    "pourquoi", "quand", "que", "quel", "quelle", "quelles", "quels", "qui", "sa", "sans",
    "ses", "seulement", "si", "sien", "son", "sont", "sous", "soyez", "sujet", "sur", "ta",
    "tandis", "tellement", "tels", "tes", "ton", "tous", "tout", "trop", "très", "tu", "voient",
    "vont", "votre", "vous", "vu", "ça", "étaient", "état", "étions", "été", "être",
    "une", "un", "chez", "etude", "etudes", "article", "articles",
}

STOPWORDS = set(ENGLISH_STOP_WORDS) | FRENCH_STOPWORDS | R_EXTRA_STOPWORDS


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"\b\w*disorder\w*\b", " ", text)  # rule in supplied R script
    text = re.sub(r"[^a-z]+", " ", text)
    return " ".join(w for w in text.split() if len(w) > 2 and w not in STOPWORDS)


def stratified_sample(df: pd.DataFrame, max_documents: int) -> pd.DataFrame:
    if len(df) <= max_documents:
        return df.copy()
    per_year = max(1, max_documents // max(1, df["year"].nunique()))
    sampled = pd.concat([
        group.sample(min(len(group), per_year), random_state=SEED)
        for _, group in df.groupby("year")
    ])
    if len(sampled) < max_documents:
        remaining = df.drop(sampled.index, errors="ignore")
        sampled = pd.concat([sampled, remaining.sample(min(len(remaining), max_documents-len(sampled)),
                                                        random_state=SEED)])
    return sampled.head(max_documents).sort_values("year")


def fit_topics(df: pd.DataFrame, n_topics: int = 4, max_features: int = 2500,
               max_documents: int = 6000):
    work = stratified_sample(df, max_documents).copy()
    work["clean_text"] = work["text"].map(normalize)
    work = work[work["clean_text"].str.len() > 20]
    min_df = 2 if len(work) < 500 else 5
    vectorizer = CountVectorizer(max_features=max_features, min_df=min_df, max_df=0.92)
    matrix = vectorizer.fit_transform(work["clean_text"])
    if matrix.shape[0] < n_topics or matrix.shape[1] < n_topics:
        raise ValueError("Not enough usable documents or terms for this number of topics.")
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=SEED,
        learning_method="batch", max_iter=15, evaluate_every=-1, n_jobs=1)
    weights = lda.fit_transform(matrix)
    terms = np.asarray(vectorizer.get_feature_names_out())
    top_words = []
    for component in lda.components_:
        idx = component.argsort()[-12:][::-1]
        top_words.append([(terms[i], float(component[i])) for i in idx])
    annual = pd.DataFrame(weights, columns=[f"Topic {i+1}" for i in range(n_topics)])
    annual["year"] = work["year"].to_numpy()
    annual = annual.groupby("year", as_index=True).mean()
    counts = work.groupby("year").size().rename("documents")
    return {"top_words": top_words, "annual": annual, "counts": counts,
            "documents": work, "weights": weights, "perplexity": float(lda.perplexity(matrix)),
            "vocabulary_size": int(matrix.shape[1])}


def topic_table(model, year: int) -> pd.DataFrame:
    annual = model["annual"]
    if year not in annual.index:
        return pd.DataFrame(columns=["Topic", "Prevalence", "Top terms"])
    rows = []
    for i, words in enumerate(model["top_words"]):
        rows.append({"Topic": f"Topic {i+1}", "Prevalence": float(annual.loc[year, f"Topic {i+1}"]),
                     "Top terms": ", ".join(w for w, _ in words[:10])})
    return pd.DataFrame(rows).sort_values("Prevalence", ascending=False)


def _umass_coherence(matrix, components, top_n: int = 10) -> float:
    """Mean UMass coherence; higher (less negative) values are better."""
    binary = (matrix > 0).astype(np.int8).tocsc()
    document_frequency = np.asarray(binary.sum(axis=0)).ravel()
    scores = []
    for component in components:
        top = component.argsort()[-top_n:][::-1]
        topic_scores = []
        for m in range(1, len(top)):
            for prior in range(m):
                denominator = max(1, int(document_frequency[top[prior]]))
                cooccurrence = int(binary[:, top[m]].multiply(binary[:, top[prior]]).sum())
                topic_scores.append(np.log((cooccurrence + 1.0) / denominator))
        scores.append(float(np.mean(topic_scores)))
    return float(np.mean(scores))


def evaluate_topic_counts(df: pd.DataFrame, k_values=range(2, 9), max_features: int = 1800,
                          max_documents: int = 3000, max_iter: int = 12) -> pd.DataFrame:
    """Evaluate candidate topic counts on one fixed full-period sample and vocabulary."""
    work = stratified_sample(df, max_documents).copy()
    work["clean_text"] = work["text"].map(normalize)
    work = work[work["clean_text"].str.len() > 20]
    min_df = 2 if len(work) < 500 else 5
    vectorizer = CountVectorizer(max_features=max_features, min_df=min_df, max_df=0.92)
    matrix = vectorizer.fit_transform(work["clean_text"])
    rows = []
    for k in k_values:
        lda = LatentDirichletAllocation(n_components=int(k), random_state=SEED,
            learning_method="batch", max_iter=max_iter, evaluate_every=-1, n_jobs=1)
        lda.fit(matrix)
        rows.append({"k": int(k), "coherence_umass": _umass_coherence(matrix, lda.components_),
                     "perplexity": float(lda.perplexity(matrix)),
                     "documents": int(matrix.shape[0]), "vocabulary": int(matrix.shape[1])})
    return pd.DataFrame(rows)
