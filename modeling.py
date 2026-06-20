from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from scipy.optimize import linear_sum_assignment

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
    "une", "un", "chez", "entre", "etude", "etudes", "article", "articles",
}

STOPWORDS = set(ENGLISH_STOP_WORDS) | FRENCH_STOPWORDS | R_EXTRA_STOPWORDS

# Translation is deliberately post-model: LDA sees the original vocabulary, while
# labels, cross-corpus matching, tables, and networks use English display terms.
FRENCH_TO_ENGLISH = {
    "adolescence": "adolescence", "adolescent": "adolescent", "adolescents": "adolescents",
    "age": "age", "ans": "years", "anorexie": "anorexia", "apprentissage": "learning",
    "apropos": "case report", "autisme": "autism",
    "cas": "cases", "charge": "care", "clinique": "clinical",
    "developpement": "development", "developpemental": "developmental",
    "ecole": "school", "enfance": "childhood", "enfant": "child", "enfants": "children",
    "education": "education", "evaluation": "assessment", "familial": "family", "famille": "family",
    "filles": "girls", "garcons": "boys", "handicap": "disability", "handicapes": "disabled people",
    "infantile": "childhood", "integration": "integration", "jeune": "young", "jeunes": "young people",
    "langage": "language", "maladie": "disease", "mentale": "mental", "mentaux": "mental",
    "medico": "medical", "normal": "normal", "parents": "parents", "pedopsychiatrie": "child psychiatry",
    "prevention": "prevention", "prise": "care", "propos": "reports",
    "psychiatrie": "psychiatry", "psychiatrique": "psychiatric",
    "psychopathologie": "psychopathology", "scolaire": "school", "sociale": "social",
    "sante": "health", "soins": "care", "traitement": "treatment", "traitements": "treatments",
    "trouble": "condition", "troubles": "conditions",
}

TOPIC_FAMILIES = [
    ("Development, school, and psychosocial context", {
        "development", "developmental", "school", "behavior", "social", "family", "parents",
        "child", "children", "adolescent", "adolescents", "adolescence", "childhood", "early",
        "risk", "problems", "group", "young", "people", "intellectual"}),
    ("Disability, learning, and cognition", {
        "disability", "disabilities", "retardation", "retarded", "learning", "reading", "language",
        "students", "performance", "cognitive", "cognition", "intellectual", "scores", "mental"}),
    ("Child psychiatry, mental health, and care", {
        "child", "children", "psychiatry", "psychiatric", "psychopathology", "mental", "health",
        "care", "treatment", "treatments", "clinical", "adolescent", "adolescents", "research"}),
    ("Clinical syndromes and diagnosis", {
        "syndrome", "syndromes", "cases", "case", "clinical", "disease", "diagnosis", "diagnostic",
        "symptoms", "normal", "infants", "patients"}),
    ("Family, services, and social environment", {
        "family", "parents", "social", "community", "services", "care", "school", "children",
        "support", "groups", "education"}),
    ("Research methods and epidemiology", {
        "research", "study", "studies", "group", "groups", "years", "scores", "prevalence",
        "risk", "population", "results", "sample"}),
]

TOPIC_COLORS = ["#2E8B57", "#3B82C4", "#A855F7", "#E67E22", "#D64B4B", "#D4A72C", "#64748B", "#14B8A6"]


def display_term(term: str) -> str:
    """English-only term used after fitting for display and semantic matching."""
    return FRENCH_TO_ENGLISH.get(str(term).lower(), str(term).lower())


def _canonical(term: str) -> str:
    term = display_term(term).replace(" ", "_")
    aliases = {
        "children": "child", "adolescents": "adolescent", "conditions": "condition",
        "disabilities": "disability", "developmental": "development", "groups": "group",
        "mentally": "mental", "psychiatric": "psychiatry", "retarded": "retardation",
        "treatments": "treatment", "young_people": "young",
    }
    return aliases.get(term, term)


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
        idx = component.argsort()[-40:][::-1]
        top_words.append([(terms[i], float(component[i])) for i in idx])
    annual = pd.DataFrame(weights, columns=[f"Topic {i+1}" for i in range(n_topics)])
    annual["year"] = work["year"].to_numpy()
    annual = annual.groupby("year", as_index=True).mean()
    counts = work.groupby("year").size().rename("documents")
    return {"top_words": top_words, "annual": annual, "counts": counts,
            "documents": work, "weights": weights, "perplexity": float(lda.perplexity(matrix)),
            "vocabulary_size": int(matrix.shape[1]), "components": lda.components_,
            "terms": terms, "matrix": matrix}


def _topic_profile(words):
    profile = defaultdict(float)
    total = sum(weight for _, weight in words) or 1.0
    for term, weight in words:
        profile[_canonical(term)] += float(weight) / total
    return profile


def _cosine_dict(left, right):
    common = set(left) | set(right)
    dot = sum(left.get(k, 0.0) * right.get(k, 0.0) for k in common)
    nl = np.sqrt(sum(v * v for v in left.values()))
    nr = np.sqrt(sum(v * v for v in right.values()))
    return float(dot / (nl * nr)) if nl and nr else 0.0


def _family_scores(profile):
    scores = []
    for _, lexicon in TOPIC_FAMILIES:
        canon = {_canonical(term) for term in lexicon}
        scores.append(sum(weight for term, weight in profile.items() if term in canon))
    return np.asarray(scores, dtype=float)


def align_topic_models(fr_model, en_model):
    """Pair independently fitted topics and assign shared names, numbers, and colors."""
    fr_profiles = [_topic_profile(words) for words in fr_model["top_words"]]
    en_profiles = [_topic_profile(words) for words in en_model["top_words"]]
    similarities = np.zeros((len(fr_profiles), len(en_profiles)))
    for i, fr in enumerate(fr_profiles):
        fs = _family_scores(fr)
        for j, en in enumerate(en_profiles):
            es = _family_scores(en)
            family_similarity = float(np.dot(fs, es) / (np.linalg.norm(fs) * np.linalg.norm(es))) if np.linalg.norm(fs) and np.linalg.norm(es) else 0.0
            similarities[i, j] = 0.30 * _cosine_dict(fr, en) + 0.70 * family_similarity
    fr_idx, en_idx = linear_sum_assignment(-similarities)
    raw_pairs = []
    used_labels = set()
    for fi, ei in zip(fr_idx, en_idx):
        combined = _family_scores(fr_profiles[fi]) + _family_scores(en_profiles[ei])
        candidates = list(np.argsort(combined)[::-1])
        family = next((x for x in candidates if x not in used_labels), candidates[0])
        used_labels.add(family)
        raw_pairs.append({"fr_index": int(fi), "en_index": int(ei),
                          "label": TOPIC_FAMILIES[family][0], "family": int(family),
                          "similarity": float(similarities[fi, ei])})
    raw_pairs.sort(key=lambda item: (item["family"], item["fr_index"]))
    for number, pair in enumerate(raw_pairs, 1):
        pair.update({"topic": number, "color": TOPIC_COLORS[(number - 1) % len(TOPIC_COLORS)]})
    return {"pairs": raw_pairs, "similarity_matrix": similarities}


def aligned_model_view(model, alignment, side: str):
    index_key = "fr_index" if side == "fr" else "en_index"
    annual = pd.DataFrame(index=model["annual"].index)
    words, meta = [], []
    for pair in alignment["pairs"]:
        raw = pair[index_key]
        name = f"Topic {pair['topic']} — {pair['label']}"
        annual[name] = model["annual"][f"Topic {raw + 1}"]
        words.append([(display_term(term), weight) for term, weight in model["top_words"][raw]])
        meta.append({**pair, "raw_index": raw, "name": name})
    return {**model, "annual_aligned": annual, "top_words_aligned": words, "topic_meta": meta}


def topic_table(model, year: int) -> pd.DataFrame:
    annual = model.get("annual_aligned", model["annual"])
    if year not in annual.index:
        return pd.DataFrame(columns=["Topic", "Prevalence", "Top terms"])
    rows = []
    words_list = model.get("top_words_aligned", model["top_words"])
    for i, words in enumerate(words_list):
        topic_name = annual.columns[i]
        rows.append({"Topic": topic_name, "Prevalence": float(annual.loc[year, topic_name]),
                     "Top terms": ", ".join(w for w, _ in words[:10])})
    return pd.DataFrame(rows).sort_values("Prevalence", ascending=False)


def topic_network_dot(model, title: str, max_nodes: int = 54, max_edges: int = 90):
    """Graphviz co-occurrence network; node communities inherit aligned LDA topics."""
    raw_to_meta = {m["raw_index"]: m for m in model["topic_meta"]}
    selected = []
    for raw_index, component in enumerate(model["components"]):
        for term_index in component.argsort()[-max(12, max_nodes // len(model["components"])) :][::-1]:
            selected.append((int(term_index), float(component[term_index]), raw_index))
    selected = sorted(selected, key=lambda x: x[1], reverse=True)
    unique = []
    seen = set()
    for item in selected:
        if item[0] not in seen:
            unique.append(item)
            seen.add(item[0])
        if len(unique) >= max_nodes:
            break
    indices = [item[0] for item in unique]
    binary = (model["matrix"][:, indices] > 0).astype(np.int8)
    freq = np.asarray(binary.sum(axis=0)).ravel().astype(float)
    cooc = (binary.T @ binary).toarray().astype(float)
    candidates = []
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            if cooc[i, j] >= 3:
                association = cooc[i, j] / np.sqrt(max(1.0, freq[i] * freq[j]))
                candidates.append((association, i, j, cooc[i, j]))
    candidates.sort(reverse=True)
    max_freq = max(freq) if len(freq) else 1.0
    lines = ["graph topics {", "graph [overlap=false, splines=true, bgcolor=white, pad=0.25];",
             "node [shape=circle, style=filled, fontname=Arial, fontsize=10, color=white, penwidth=1.2];",
             "edge [color=\"#64748B55\"];", f'label="{title}";', "labelloc=t;", "fontsize=18;"]
    for local, (term_index, _, raw_index) in enumerate(unique):
        meta = raw_to_meta[raw_index]
        label = display_term(model["terms"][term_index]).replace('"', "'")
        size = 0.35 + 0.65 * np.sqrt(freq[local] / max_freq)
        tooltip = f"{label}: {int(freq[local])} documents — Topic {meta['topic']}"
        lines.append(f'n{local} [label="{label}", fillcolor="{meta["color"]}", width={size:.2f}, height={size:.2f}, tooltip="{tooltip}"];')
    chosen = candidates[:max_edges]
    values = [x[0] for x in chosen] or [1.0]
    low, high = min(values), max(values)
    for association, i, j, count in chosen:
        scaled = 0.7 + 3.3 * (association - low) / (high - low) if high > low else 1.5
        lines.append(f'n{i} -- n{j} [penwidth={scaled:.2f}, weight={1 + int(association * 12)}, tooltip="{int(count)} co-occurrences"];')
    lines.append("}")
    return "\n".join(lines)


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
