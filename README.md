# Child Psychiatry Topic Explorer

Streamlit application for comparing the annual evolution of topics in the
francophone and anglophone child-psychiatry literature.

## Run on: https://child-psychiatry-topic-explore-6j3jejfsuqfz3taiy49jpt.streamlit.app

The repository contains a cleaned, compressed historical corpus so the cloud
deployment is self-contained. The application can also query PubMed through
NCBI E-utilities or accept additional RIS files uploaded by the user. Local
source discovery remains available when the original RIS exports are present.

## Reproducibility

- Search equations are stored in `config.py`.
- Random seed: 1234 (matching the supplied R code).
- Default number of topics: 3. This was selected over k=2–8 using the maximum
  mean within-corpus standardized UMass coherence across both full-period
  corpora and is supported by the main perplexity elbow. The original R value
  (k=4) remains available through the slider.
- Topic models are fitted independently by linguistic corpus.
- French terms are translated into English only after model estimation, so
  translation cannot change the learned co-occurrences. The source records may
  themselves be bilingual.
- Independently fitted topics are matched one-to-one from their weighted,
  translated semantic profiles. Matched pairs receive the same number, label,
  and color in both corpora; the app also reports the match score and original
  source-topic numbers.
- Topic prevalence is the annual mean document-topic weight.
- Lexical networks use document frequency for node size and normalized
  within-document co-occurrence for edge width.

Pascal/Francis is supported through historical RIS imports because no stable
public programmatic API is available for the legacy database.

## Streamlit Community Cloud

Deploy `app.py` from the repository root with the standard Python runtime.
No secret is required for the bundled historical analysis; an optional NCBI
API key can be configured separately for higher PubMed request limits.
