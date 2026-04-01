# Substance Abuse Detection AI
### NSF NRT Research-A-Thon 2026 — Challenge 1, Track B: Data Intelligence + Decision Support
**UMKC · Hosted by the NSF Research Traineeship Program**

---

## Team

| Name | Role |
|------|------|
| **Tony Nguyen** | Model Builder — LLM classifier, embeddings, RAG pipeline |
| **Tina Nguyen** | Pipeline & Story — Data cleaning, temporal analysis, explainability |
| **Joe Doan** | Pipeline & Story — Data cleaning, temporal analysis, explainability |

---

## What We're Building

An end-to-end AI pipeline that ingests messy social text, extracts substance-abuse risk signals, reasons about patterns over time, and surfaces explainable, privacy-preserving insights for public health decision-makers.

**Core tasks:**
1. **Ingestion & preprocessing** — clean text, apply slang lexicon, strip PII, deduplicate
2. **Risk signal detection** — three parallel methods (rule-based, embedding, LLM) with benchmarked comparison
3. **Temporal & behavioral analysis** — spike detection, rolling signal rates, topic clustering
4. **Explainability layer** — RAG-powered summaries for public health analysts, always aggregate

---

## Architecture

```
Raw Data → Preprocessing → Detection (3-way) → Temporal Analysis → Explainability → Dashboard
```

| Layer | What it does | Key tools |
|-------|-------------|----------|
| **1 — Ingestion & Preprocessing** | HTML clean, PII scrub, slang normalization, dedup | pandas, spaCy, regex |
| **2 — Risk Signal Detection** | Rule-based · Embedding cosine-similarity · LLM classifier | Sentence-BERT, Claude Haiku API |
| **3 — Temporal & Behavioral** | Daily/weekly binning, z-score spike detection, topic clustering | pandas, K-Means / HDBSCAN |
| **4 — Explainability (RAG)** | Retrieve top-k supporting posts → LLM 3-sentence analyst summaries | FAISS / Chroma, Claude API |

**Detection methods compared:**

| Method | Description |
|--------|-------------|
| Rule-based | Regex + slang dictionary for substance mentions and distress phrases |
| Embedding-based | Sentence-BERT cosine similarity against labeled high-risk seed examples |
| LLM-based | Gemini Flash classifies risk level + extracts evidence spans + generates analyst summaries |

---

## Data Sources

| Source | Purpose |
|--------|---------|
| Instructor-provided forum dataset | Primary social signal (anonymized) |
| [KUC Hackathon / UCI Drug Reviews](https://www.kaggle.com/datasets/jessicali9530/kuc-hackathon-winter-2018) | Labeled seed examples + risk label training |
| [CDC Drug Overdose API](https://data.cdc.gov/resource/xkb8-kh2a.json) | Ground-truth overdose deaths for temporal correlation |
| NSDUH (SAMHSA) | Population disorder prevalence — normalize social signal |
| NIDA Summary Tables | National overdose death rates per 100k |

---

## Repository Layout

```
data/
  raw/                             # source CSVs (not committed)
  processed/                       # pipeline outputs
  processed/narrative/             # temporal analysis outputs + Plotly figures
scripts/
  fetch_cdc_data.py                # download CDC overdose data
  fetch_nsduh.py                   # NSDUH disorder prevalence rates
  fetch_nida_summary.py            # NIDA overdose death rates
src/
  processing/
    process_drug_reviews.py        # Kaggle corpus → posts_classified.csv + seed bank
    preprocess_posts.py            # text preprocessing (clean → PII scrub → slang → signals)
  classifiers/
    rule_based_classifier.py       # Layer 1: 3-layer scored rule-based classifier
    embedding_classifier.py        # Layer 2: sentence-BERT similarity + clustering
    llm_classifier.py              # Layer 3: Gemini Flash + RAG spike summaries
    ensemble.py                    # Weighted vote fusion of all three methods
  agents/
    signal_pipeline.py             # CDC alignment + NSDUH normalization + cross-correlation
    rag_pipeline.py                # Advanced RAG + cross-encoder reranking for analyst reports
  utils/
    narrative_evolution.py         # Temporal drift, topic tracking, UMAP, early-warning alerts
  eval/
    generate_comparison.py         # Standalone method comparison with proxy accuracy/F1
  app/
    dashboard.py                   # Streamlit dashboard (7 tabs)
```

---

## Quickstart

```powershell
# 1. Activate venv
.\venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 3. Fetch public health data
python scripts/fetch_cdc_data.py
python scripts/fetch_nsduh.py
python scripts/fetch_nida_summary.py

# 4. Process drug review corpus (requires drugsComTrain/Test_raw.csv in data/raw/)
python src/processing/process_drug_reviews.py

# 5. Preprocess posts
python src/processing/preprocess_posts.py

# 6. Run rule-based classifier (Layer 1)
python src/classifiers/rule_based_classifier.py

# 7. Run embedding classifier (Layer 2 — clustering + similarity scoring)
python src/classifiers/embedding_classifier.py

# 8. Run ensemble fusion
python src/classifiers/ensemble.py

# 9. Run full signal pipeline (CDC alignment + cross-correlation)
python src/agents/signal_pipeline.py

# 10. Run narrative evolution analysis (drift, topics, UMAP, alerts)
python src/utils/narrative_evolution.py

# 11. Generate method comparison with accuracy/F1 metrics
python src/eval/generate_comparison.py

# 12. (Optional) Generate RAG analyst reports — requires GOOGLE_API_KEY
$env:GOOGLE_API_KEY = "<your-key>"
python src/agents/rag_pipeline.py

# 13. Launch dashboard
streamlit run src/app/dashboard.py
```

---

## Key Dates

| Milestone | Date |
|-----------|------|
| Team Registration | **March 30, 2026** ⚠️ |
| Project Submission | April 6, 2026 (noon) |
| Finalist Announcement | April 8, 2026 |
| Final Demo | April 10, 2026 · 9AM–12PM · SU 401 |

---

## Ethics & Privacy

- No individual identification — PII scrubbed before any model sees text
- Population-level insights only; outputs always aggregate
- Public and instructor-provided datasets exclusively
- All summaries framed for public health decision-makers, not individual surveillance

---

## Contact

- **AI Challenge Lead:** Dr. Yugyung Lee — leeyu@umkc.edu
- **Engineering Challenge Lead:** Dr. Mostafizur Rahman — rahmanmo@umsystem.edu
