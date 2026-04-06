# Substance Abuse Detection AI
### NSF NRT Research-A-Thon 2026 — Challenge 1, Track B: Data Intelligence + Decision Support
**UMKC · Hosted by the NSF Research Traineeship Program**

---

## Team

| Name | Role |
|------|------|
| **Tony Nguyen** | Model Builder (LLM classifier, embedding, RAG pipeline) — Fine-tuned BERT classifier, embeddings, cluster quality metrics, ensemble fusion |
| **Daniel Evans** | Pipeline & Story — Temporal analysis metrics, evaluation module, ROC/confusion matrix figures |
| **Joel Vinas** | Pipeline & Story — Dashboard decision support, Temporal Analysis, intervention engine, testing framework |
| **Tina Nguyen** | Pipeline & Story — Final report, human-in-the-loop evaluation, pipeline & explainability |

---

## What We're Building

An end-to-end AI pipeline that ingests messy social text, extracts substance-abuse risk signals, reasons about patterns over time, and surfaces explainable, privacy-preserving insights for public health decision-makers.

**Core tasks:**
1. **Ingestion & preprocessing** — clean text, apply slang lexicon (extended with Erowid NMF terms), strip PII, deduplicate
2. **Risk signal detection** — four parallel methods (rule-based, embedding, LLM, fine-tuned BERT) with benchmarked comparison
3. **Temporal & behavioral analysis** — spike detection, rolling signal rates, topic clustering, Erowid-powered spillover detection
4. **Explainability layer** — RAG-powered summaries for public health analysts, grounded with Erowid community-report context

---

## Architecture

```
Raw Data -> Preprocessing -> Detection (4-way) -> Temporal Analysis -> Explainability -> Dashboard
```

| Layer | What it does | Key tools |
|-------|-------------|----------|
| **1 — Ingestion & Preprocessing** | HTML clean, PII scrub, slang normalization (+ Erowid NMF extensions), dedup | pandas, spaCy, regex |
| **2 — Risk Signal Detection** | Rule-based (+ Erowid harm boost) · Embedding cosine-similarity · LLM classifier · Fine-Tuned BERT | Sentence-BERT, DistilBERT, Gemini Flash |
| **3 — Temporal & Behavioral** | Daily/weekly binning, z-score spike detection, Erowid spillover detection, topic clustering | pandas, K-Means / HDBSCAN |
| **4 — Explainability (RAG)** | Retrieve top-k supporting posts -> LLM analyst summaries grounded with Erowid NMF topic context | FAISS / Chroma, Gemini Flash |

**Detection methods compared (held-out test set, 2,000 posts):**

| Method | Description | Accuracy | Macro-F1 | ROC-AUC |
|--------|-------------|----------|-----------|---------|
| Rule-based | Regex + slang dictionary + Erowid NMF harm-overlap weight boost | 0.495 | 0.303 | 0.504 |
| Embedding-based | Sentence-BERT cosine similarity against labeled high-risk seed examples | 0.325 | 0.295 | 0.490 |
| LLM-based | Gemini Flash classifies risk level + extracts evidence spans + generates analyst summaries | — | — | — |
| **Fine-Tuned BERT** | DistilBERT fine-tuned on 41k posts with ensemble pseudo-labels | 0.404 | 0.280 | 0.481 |
| **Ensemble (winner)** | Weighted vote: Rule(0.20) + Embed(0.30) + LLM(0.40) + FineTuned(0.10) | **0.404** | **0.304** | **0.506** |

**Temporal metrics:**

| Metric | Value | Meaning |
|--------|-------|---------|
| Overall MRR | 0.1508 | Relevant alerts surface near top of ranked list during CDC spikes |
| Opioid MRR | 0.2984 | Strongest alert prioritisation for opioid events |
| Opioid detection lag | −35½ months (median) | Social signal precedes CDC spike by ~11 months — early warning confirmed |

---

## Data Sources

| Source | Purpose |
|--------|---------|
| [KUC Hackathon / UCI Drug Reviews](https://www.kaggle.com/datasets/jessicali9530/kuc-hackathon-winter-2018) | Labeled seed examples + risk label training (41,934 posts) |
| [CDC Drug Overdose API](https://data.cdc.gov/resource/xkb8-kh2a.json) | Ground-truth overdose deaths for temporal correlation |
| NSDUH (SAMHSA) | Population disorder prevalence — normalize social signal |
| NIDA Summary Tables | National overdose death rates per 100k |
| [Erowid Experience Vaults](https://erowid.org/experiences/) via [erowid-lsa](https://github.com/Monkeyanator/erowid-lsa) | Aggregate NMF topic signals: substance harm vocabulary, semantic similarity graph, RAG context (signal discovery only — not ground truth) |

> **PI approval:** Erowid dataset approved for aggregate signal-discovery use. All outputs are population-level; no individual reports are stored or surfaced. Integrated with CDC/NIDA data per PI guidance.

---

## Repository Layout

```
data/
  raw/
    erowid-lsa-repo/experiences/    # scraped Erowid experience reports (166 substance dirs)
    erowid_posts.csv                # classified Erowid posts (schema-compatible with posts_classified.csv)
  processed/
    erowid_substance_profiles.json  # Erowid NMF topic profiles per substance (19 substances)
    erowid_substance_similarity.json # pairwise NMF cosine similarity between substances
    erowid_spillover.csv            # co-spiking substance pairs from spillover detection
    narrative/                      # temporal analysis outputs + Plotly figures
scripts/
  fetch_cdc_data.py                 # download CDC overdose data
  fetch_nsduh.py                    # NSDUH disorder prevalence rates
  fetch_nida_summary.py             # NIDA overdose death rates
  fetch_erowid.py                   # download Erowid LSA source files + run scraper
  process_erowid_lsa.py             # NMF analysis -> erowid_substance_profiles.json + similarity JSON
src/
  processing/
    process_drug_reviews.py         # Kaggle corpus -> posts_classified.csv + seed bank
    preprocess_posts.py             # text preprocessing (clean -> PII scrub -> slang -> signals)
    process_erowid.py               # Erowid HTML reports -> erowid_posts.csv (BS4 parse, PII scrub, risk label)
    process_erowid_lsa.py           # Erowid reports -> erowid_posts.csv (NMF/LSA variant)
  classifiers/
    rule_based_classifier.py        # Layer 1: 3-layer scored rule-based + Erowid NMF weight boost
    embedding_classifier.py         # Layer 2: sentence-BERT similarity + clustering
    llm_classifier.py               # Layer 3: Gemini Flash + RAG spike summaries
    finetuned_classifier.py         # Layer 4: DistilBERT fine-tuned on pseudo-labels
    ensemble.py                     # Weighted vote fusion of all four methods
  agents/
    signal_pipeline.py              # CDC alignment + NSDUH normalization + cross-correlation + Erowid source merge + spillover
    rag_pipeline.py                 # Advanced RAG + cross-encoder reranking + Erowid NMF context injection
    intervention_engine.py          # Recommendation generation from spike events
  utils/
    narrative_evolution.py          # Temporal drift, topic tracking, UMAP, early-warning alerts
  eval/
    generate_comparison.py          # Standalone method comparison with proxy accuracy/F1
    cluster_metrics.py              # Silhouette Score, NDCG, Perplexity for cluster quality
    temporal_metrics.py             # MRR + Detection Lag vs. CDC spike events -> temporal_metrics.json
    evaluation_report.py            # Held-out test set: Accuracy/F1/AUC + Plotly figures -> eval_figures/
    summary_metrics.py              # ROUGE-L, BERTScore, Faithfulness for RAG summaries
  tests/
    test_metamorphic.py             # Slang substitution robustness tests
    test_safety_guards.py           # PII regression tests
  app/
    dashboard.py                    # Streamlit dashboard (8 tabs)
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

# 4. Fetch and process Erowid experience corpus (signal enrichment — run once)
python scripts/fetch_erowid.py                  # git clone Erowid mirror -> data/raw/erowid-lsa-repo/
python src/processing/process_erowid.py         # parse HTML reports, scrub PII, label risk -> erowid_posts.csv
python src/processing/process_erowid_lsa.py     # (alt) classify reports via LSA variant -> erowid_posts.csv
python scripts/process_erowid_lsa.py            # NMF analysis -> substance profiles + similarity JSON

# 5. Process drug review corpus (requires drugsComTrain/Test_raw.csv in data/raw/)
python src/processing/process_drug_reviews.py

# 6. Preprocess posts (slang lexicon extended with Erowid NMF terms)
python src/processing/preprocess_posts.py

# 7. Run rule-based classifier (Layer 1 — substance weights boosted by Erowid NMF harm signals)
python src/classifiers/rule_based_classifier.py

# 8. Run embedding classifier (Layer 2 — clustering + similarity scoring)
python src/classifiers/embedding_classifier.py

# 9. Run LLM classifier (Layer 3 — Gemini Flash risk classification + evidence spans)
$env:GOOGLE_API_KEY = "<your-key>"
python src/classifiers/llm_classifier.py

# 10. Run fine-tuned BERT classifier (Layer 4)
python src/classifiers/finetuned_classifier.py

# 11. Run ensemble fusion
python src/classifiers/ensemble.py

# 12. Run full signal pipeline (CDC alignment + cross-correlation + Erowid spillover detection)
python src/agents/signal_pipeline.py

# 13. Run narrative evolution analysis (drift, topics, UMAP, alerts)
python src/utils/narrative_evolution.py

# 14. Compute cluster quality metrics (Silhouette, NDCG, Perplexity)
python src/eval/cluster_metrics.py

# 15. Generate method comparison with accuracy/F1 metrics
python src/eval/generate_comparison.py

# 16. Compute temporal metrics (MRR, Detection Lag vs. CDC spikes)
python src/eval/temporal_metrics.py

# 17. Generate held-out test-set evaluation report + Plotly figures
python src/eval/evaluation_report.py

# 18. (Optional) Generate RAG analyst reports — requires GOOGLE_API_KEY
$env:GOOGLE_API_KEY = "<your-key>"
python src/agents/rag_pipeline.py               # summaries include Erowid NMF community context

# 19. Launch dashboard
streamlit run src/app/dashboard.py
```

---

## Key Dates

| Milestone | Date |
|-----------|------|
| Team Registration | **March 30, 2026** |
| Project Submission | April 6, 2026 (noon) |
| Finalist Announcement | April 8, 2026 |
| Final Demo | April 10, 2026 · 9AM–12PM · SU 401 |

---

## Ethics & Privacy

- No individual identification — PII scrubbed before any model sees text
- Population-level insights only; outputs always aggregate
- Public and instructor-provided datasets exclusively
- Erowid data used as aggregate signal discovery only (PI-approved); no individual reports stored or surfaced
- All summaries framed for public health decision-makers, not individual surveillance

---

## Contact

- **AI Challenge Lead:** Dr. Yugyung Lee — leeyu@umkc.edu
- **Engineering Challenge Lead:** Dr. Mostafizur Rahman — rahmanmo@umsystem.edu
