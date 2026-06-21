from pathlib import Path
import sys

APP_DIR = Path(__file__).resolve().parent
VENDORED = APP_DIR / ".packages"
if VENDORED.exists():
    sys.path.insert(0, str(VENDORED))

import pandas as pd
import streamlit as st

from config import DEFAULT_END_YEAR, DEFAULT_START_YEAR, DEFAULT_TOPICS, PASCAL_FRANCIS_QUERY, pubmed_query
from corpus import clean_corpus, fetch_pubmed, load_historical, parse_ris
from modeling import fit_topics, topic_table

st.set_page_config(page_title="Child Psychiatry Topic Explorer", page_icon="🧭", layout="wide")
st.title("Child Psychiatry Topic Explorer")
st.caption("Annual topic trajectories in francophone and anglophone scientific literature")


@st.cache_data(show_spinner="Reading historical RIS corpora…")
def historical_data():
    return load_historical(APP_DIR)


@st.cache_resource(show_spinner="Fitting the topic model…")
def cached_model(serialized: str, label: str, topics: int, max_docs: int):
    frame = pd.read_json(serialized, orient="split")
    return fit_topics(frame, n_topics=topics, max_documents=max_docs)


if "corpus" not in st.session_state:
    st.session_state.corpus = historical_data()

with st.sidebar:
    st.header("Analysis settings")
    n_topics = st.slider("Number of topics", 2, 8, DEFAULT_TOPICS)
    max_docs = st.slider("Maximum documents per corpus", 500, 12000, 6000, 500)
    st.caption("Default k = 3 from full-period bicorpus coherence + perplexity elbow · seed = 1234")

tabs = st.tabs(["Topic evolution", "Corpus & retrieval", "Method"])

with tabs[0]:
    data = clean_corpus(st.session_state.corpus)
    if data.empty:
        st.warning("No corpus is loaded. Import a RIS file in the next tab.")
    else:
        valid_years = sorted(data["year"].unique().tolist())
        year_min, year_max = int(min(valid_years)), int(max(valid_years))
        selected_range = st.slider("Modeling period", year_min, year_max, (year_min, year_max))
        selected_year = st.select_slider("Year to inspect", options=list(range(selected_range[0], selected_range[1] + 1)),
                                         value=selected_range[1])
        filtered = data[data["year"].between(*selected_range)]
        cols = st.columns(2)
        for column, label in zip(cols, ["Francophone", "Anglophone"]):
            with column:
                st.subheader(label)
                subset = filtered[filtered["language"] == label]
                st.metric("Usable records", f"{len(subset):,}")
                if len(subset) < max(30, n_topics * 5):
                    st.info("Not enough records in this period.")
                    continue
                try:
                    payload = subset[["id", "title", "abstract", "year", "language", "source", "doi", "journal", "text"]].to_json(orient="split")
                    model = cached_model(payload, label, n_topics, max_docs)
                    st.line_chart(model["annual"], height=280)
                    table = topic_table(model, selected_year)
                    if table.empty:
                        st.caption(f"No sampled document for {selected_year}.")
                    else:
                        st.dataframe(table, hide_index=True, use_container_width=True,
                                     column_config={"Prevalence": st.column_config.ProgressColumn(
                                         "Prevalence", min_value=0.0, max_value=1.0, format="%.3f")})
                    with st.expander("Model diagnostics"):
                        st.write({"documents modeled": len(model["documents"]),
                                  "vocabulary": model["vocabulary_size"],
                                  "perplexity": round(model["perplexity"], 1)})
                import traceback
                except Exception:
                    st.code(traceback.format_exc())

with tabs[1]:
    st.subheader("Loaded corpus")
    corpus = clean_corpus(st.session_state.corpus)
    summary = (corpus.groupby("language").agg(records=("id", "size"), first_year=("year", "min"),
               last_year=("year", "max"), abstracts=("abstract", lambda x: (x.str.len() > 0).sum()))
               .reset_index()) if not corpus.empty else pd.DataFrame()
    st.dataframe(summary, hide_index=True, use_container_width=True)

    uploaded = st.file_uploader("Add one or more RIS exports", type=["ris"], accept_multiple_files=True)
    upload_label = st.radio("Assign uploaded files to", ["Francophone", "Anglophone"], horizontal=True)
    if uploaded and st.button("Import RIS files"):
        additions = [parse_ris(item, upload_label) for item in uploaded]
        st.session_state.corpus = clean_corpus(pd.concat([corpus, *additions], ignore_index=True))
        st.cache_resource.clear()
        st.success("RIS records imported and deduplicated.")
        st.rerun()

    st.divider()
    st.subheader("Update from PubMed")
    c1, c2, c3 = st.columns(3)
    language = c1.selectbox("Language", ["English", "French"])
    start_year = c2.number_input("From", 1945, DEFAULT_END_YEAR, 2016)
    end_year = c3.number_input("To", int(start_year), DEFAULT_END_YEAR, DEFAULT_END_YEAR)
    query = st.text_area("PubMed query", pubmed_query(language, int(start_year), int(end_year)), height=150)
    email = st.text_input("Contact email required by NCBI", placeholder="name@institution.org")
    api_key = st.text_input("NCBI API key (optional)", type="password")
    limit = st.number_input("Maximum records for this update", 20, 5000, 500, 20)
    if st.button("Retrieve from PubMed", type="primary"):
        if "@" not in email:
            st.error("Enter a valid contact email before querying NCBI.")
        else:
            with st.spinner("Querying PubMed and retrieving XML records…"):
                live = fetch_pubmed(query, "Francophone" if language == "French" else "Anglophone",
                                    email, api_key, int(limit))
            st.session_state.corpus = clean_corpus(pd.concat([corpus, live], ignore_index=True))
            st.cache_resource.clear()
            st.success(f"{len(live)} PubMed records added after cleaning.")
            st.rerun()

    with st.expander("Pascal/Francis legacy equation"):
        st.code(PASCAL_FRANCIS_QUERY, language=None)
        st.info("Use a RIS export above. The legacy database is not queried automatically because a stable public API is unavailable.")

    if not corpus.empty:
        st.download_button("Download cleaned corpus (CSV)", corpus.to_csv(index=False).encode("utf-8"),
                           "child_psychiatry_corpus.csv", "text/csv")

with tabs[2]:
    st.markdown("""
### Reproducible pipeline

1. Retrieve records with the supplied MeSH equations and language/date filters.
2. Merge PubMed and imported Pascal/Francis RIS records, then deduplicate by DOI or normalized title and year.
3. Analyze title plus abstract; normalize case and accents; remove English/French stopwords and the additional exclusions from the supplied R script, including terms containing *disorder*.
4. Fit Latent Dirichlet Allocation independently in each linguistic corpus. The original R value was *k* = 4; the application default is now *k* = 3, selected over *k* = 2–8 by the maximum mean within-corpus standardized UMass coherence across the two full-period corpora. The largest average perplexity improvement also occurs from *k* = 2 to *k* = 3.
5. Estimate annual prevalence as the mean posterior topic weight among documents published in each year.

The curves describe themes inside each corpus; topic numbers are not assumed to be semantically identical across languages. A substantive cross-language interpretation should compare their top terms and trajectories, not merely “Topic 1” with “Topic 1”.
""")
    validation_path = APP_DIR / "topic_count_validation.csv"
    if validation_path.exists():
        validation = pd.read_csv(validation_path)
        st.subheader("Selection of the default topic count")
        st.info("Corpus-specific coherence optima differ: k = 3 for the francophone corpus and k = 8 for the anglophone corpus. The common default k = 3 is the best standardized bicorpus compromise and coincides with the main perplexity elbow.")
        chart = validation.pivot(index="k", columns="corpus", values="coherence_umass")
        st.line_chart(chart)
        st.dataframe(validation[["corpus", "k", "coherence_umass", "perplexity"]],
                     hide_index=True, use_container_width=True)
    st.caption("NCBI disclaimer: PubMed abstracts may be protected by copyright. Users are responsible for compliant reuse and redistribution.")
