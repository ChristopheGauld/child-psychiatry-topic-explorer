from __future__ import annotations

import io
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import BinaryIO, Union

import pandas as pd
import requests

FIELDS = ["id", "title", "abstract", "year", "language", "source", "doi", "journal"]


def _text(value) -> str:
    return " ".join(str(value or "").split())


def parse_ris(source: Union[BinaryIO, io.BytesIO, str, Path], corpus_label: str) -> pd.DataFrame:
    if hasattr(source, "read"):
        raw = source.read()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
    else:
        raw = Path(source).read_text(encoding="utf-8", errors="replace")

    records, record, last_tag = [], {}, None
    repeatable = {"AU", "KW"}
    for line in raw.splitlines():
        if len(line) >= 6 and line[2:6] == "  - ":
            tag, value = line[:2], line[6:].strip()
            last_tag = tag
            if tag == "TY":
                record = {"TY": value}
            elif tag == "ER":
                if record:
                    records.append(record)
                record, last_tag = {}, None
            elif tag in repeatable:
                record.setdefault(tag, []).append(value)
            elif tag in record and value:
                record[tag] = f"{record[tag]} {value}"
            else:
                record[tag] = value
        elif record and last_tag and line.strip():
            if isinstance(record.get(last_tag), list):
                record[last_tag][-1] += " " + line.strip()
            else:
                record[last_tag] = (record.get(last_tag, "") + " " + line.strip()).strip()
    if record:
        records.append(record)

    rows = []
    for i, rec in enumerate(records):
        year_raw = str(rec.get("PY") or rec.get("Y1") or rec.get("DA") or "")
        digits = "".join(ch for ch in year_raw[:10] if ch.isdigit())
        year = int(digits[:4]) if len(digits) >= 4 else None
        title = _text(rec.get("TI") or rec.get("T1"))
        abstract = _text(rec.get("AB") or rec.get("N2"))
        rid = _text(rec.get("AN") or rec.get("DO") or f"{corpus_label}-{i}")
        rows.append({
            "id": rid, "title": title, "abstract": abstract, "year": year,
            "language": corpus_label, "source": _text(rec.get("DP") or "RIS"),
            "doi": _text(rec.get("DO")), "journal": _text(rec.get("T2") or rec.get("JO")),
        })
    return clean_corpus(pd.DataFrame(rows, columns=FIELDS))


def clean_corpus(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=FIELDS)
    out = df.copy()
    for col in FIELDS:
        if col not in out:
            out[col] = ""
    out["year"] = pd.to_numeric(out["year"], errors="coerce")
    out = out.dropna(subset=["year"])
    out["year"] = out["year"].astype(int)
    out["text"] = (out["title"].fillna("") + ". " + out["abstract"].fillna("")).str.strip()
    out = out[out["text"].str.len() >= 30]
    key = out["doi"].str.lower().str.strip()
    fallback = (out["title"].str.lower().str.replace(r"\W+", " ", regex=True).str.strip()
                + "|" + out["year"].astype(str))
    out["dedup_key"] = key.where(key.str.len() > 5, fallback)
    out = out.drop_duplicates("dedup_key").drop(columns="dedup_key")
    return out.reset_index(drop=True)


def load_historical(base_dir: Path) -> pd.DataFrame:
    bundled = base_dir / "data" / "historical_corpus.csv.gz"
    if bundled.exists():
        return clean_corpus(pd.read_csv(bundled, compression="gzip", low_memory=False))
    candidates = [
        (base_dir.parent / "1 France" / "Data" / "French_Child.ris", "Francophone"),
        (base_dir.parent / "2 Anglais" / "pubmed-ChildPsych-set.ris", "Anglophone"),
    ]
    frames = [parse_ris(path, label) for path, label in candidates if path.exists()]
    return clean_corpus(pd.concat(frames, ignore_index=True)) if frames else pd.DataFrame(columns=FIELDS)


def fetch_pubmed(query: str, language_label: str, email: str, api_key: str = "",
                 max_records: int = 500) -> pd.DataFrame:
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    common = {"db": "pubmed", "tool": "hp_psy_topic_explorer", "email": email}
    if api_key:
        common["api_key"] = api_key
    search = requests.get(f"{base}/esearch.fcgi", params={**common, "term": query,
        "retmode": "json", "retmax": max_records}, timeout=60)
    search.raise_for_status()
    ids = search.json().get("esearchresult", {}).get("idlist", [])
    rows = []
    for start in range(0, len(ids), 200):
        batch = ids[start:start + 200]
        response = requests.post(f"{base}/efetch.fcgi", data={**common, "id": ",".join(batch),
            "retmode": "xml"}, timeout=120)
        response.raise_for_status()
        root = ET.fromstring(response.content)
        for article in root.findall(".//PubmedArticle"):
            med = article.find("MedlineCitation")
            art = med.find("Article") if med is not None else None
            if med is None or art is None:
                continue
            pmid = _text(med.findtext("PMID"))
            title = _text("".join(art.find("ArticleTitle").itertext()) if art.find("ArticleTitle") is not None else "")
            abst = " ".join(_text("".join(x.itertext())) for x in art.findall("Abstract/AbstractText"))
            journal = _text(art.findtext("Journal/Title"))
            year = art.findtext("Journal/JournalIssue/PubDate/Year") or art.findtext("ArticleDate/Year")
            if not year:
                medline_date = art.findtext("Journal/JournalIssue/PubDate/MedlineDate") or ""
                year = medline_date[:4]
            doi = ""
            for loc in art.findall("ELocationID"):
                if loc.attrib.get("EIdType") == "doi":
                    doi = _text(loc.text)
            rows.append({"id": pmid, "title": title, "abstract": abst, "year": year,
                "language": language_label, "source": "PubMed live", "doi": doi,
                "journal": journal})
        if start + 200 < len(ids):
            time.sleep(0.11 if api_key else 0.35)
    return clean_corpus(pd.DataFrame(rows, columns=FIELDS))
