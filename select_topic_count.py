import json
from pathlib import Path

import numpy as np
import pandas as pd

from corpus import load_historical
from modeling import evaluate_topic_counts

APP_DIR = Path(__file__).resolve().parent
OUTPUT_CSV = APP_DIR / "topic_count_validation.csv"
OUTPUT_JSON = APP_DIR / "topic_count_validation.json"

corpus = load_historical(APP_DIR)
frames = []
for label in ["Francophone", "Anglophone"]:
    print(f"Evaluating {label}...", flush=True)
    result = evaluate_topic_counts(corpus[corpus["language"] == label],
                                   max_documents=12000, max_features=2500, max_iter=12)
    result.insert(0, "corpus", label)
    frames.append(result)

scores = pd.concat(frames, ignore_index=True)
scores["coherence_z"] = scores.groupby("corpus")["coherence_umass"].transform(
    lambda x: (x - x.mean()) / (x.std(ddof=0) or 1.0))
scores["perplexity_improvement"] = scores.groupby("corpus")["perplexity"].transform(
    lambda x: -x.pct_change())
common = scores.groupby("k", as_index=False).agg(
    mean_coherence_z=("coherence_z", "mean"),
    mean_coherence=("coherence_umass", "mean"),
    mean_perplexity=("perplexity", "mean"),
)
best_k = int(common.sort_values(["mean_coherence_z", "k"], ascending=[False, True]).iloc[0]["k"])
scores.to_csv(OUTPUT_CSV, index=False)
payload = {"recommended_k": best_k, "selection_rule": "maximum mean within-corpus standardized UMass coherence",
           "candidate_k": list(range(2, 9)), "seed": 1234,
           "sampling": "all eligible documents over the full historical period",
           "per_corpus": scores.to_dict(orient="records"), "common": common.to_dict(orient="records")}
OUTPUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(scores.to_string(index=False))
print("\nCommon scores:\n", common.to_string(index=False))
print(f"\nRECOMMENDED_K={best_k}")
