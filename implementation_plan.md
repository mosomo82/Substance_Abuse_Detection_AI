# NSF NRT Challenge 1 — Enhanced Submission Plan
## Team: Joel Vinas (M3), Daniel Evans (M2), Tony Nguyen (M1), Tina Nguyen (M4)
## Submission Deadline: April 6, 2026 (12:00 PM) | Target Completion: April 5, 2026 (EOD)

---

## Context

The current codebase is a strong Track B (Data Intelligence & Decision Support) submission with:
- 3-method classifier (rule-based + embedding + LLM) + ensemble on 41,934 posts
- Temporal analysis: drift detection, 8-topic tracking, CDC lead-lag correlation
- 7-tab Streamlit dashboard with RAG analyst summaries
- Privacy-preserving pipeline (PII scrubbing, population-level outputs only)

**Judging: Technical quality (40%) / Innovation (30%) / Impact (20%) / Communication (10%)**

The enhancements below are structured around the three challenge task areas and a specialized testing framework that targets the **Innovation (30%)** score.

---

## System Architecture (Enhanced)

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                        DATA INGESTION & PREPROCESSING                            │
│   Raw Posts → Cleaning → PII Scrub → Slang Norm → posts_preprocessed.csv        │
│   Drug Reviews → Heuristic Labels → Seed Bank (posts_classified.csv)            │
└────────────────────────────────────┬─────────────────────────────────────────────┘
                                     │
                     ┌───────────────▼───────────────┐
                     │       RISK SIGNAL DETECTION    │
                     │  (4 Parallel Methods)          │
                     ├───────────┬───────────┬────────┤
                     │ Rule-Based│ Embedding │  LLM   │  ◄── NEW ──►  Fine-Tuned BERT
                     │           │           │        │              (finetuned_
                     │           │           │        │              classifier.py)
                     └───────────┴─────┬─────┴────────┘
                                       │
                     ┌─────────────────▼─────────────────┐
                     │         ENSEMBLE FUSION            │
                     │  Rule(0.25)+Embed(0.30)+           │
                     │  LLM(0.35)+FineTuned(0.10)        │
                     └──────────────┬────────────────────┘
                                    │
     ┌──────────────────────────────┼──────────────────────────────┐
     │                              │                              │
     ▼                              ▼                              ▼
┌────────────────────────┐  ┌───────────────────────┐  ┌──────────────────────────┐
│  TASK 1                │  │  TASK 2               │  │  TASK 3                  │
│  Embedding Trend       │  │  Temporal Analysis    │  │  Dashboard & Decision    │
│  Discovery             │  │  Risk Signals         │  │  Support                 │
│                        │  │  Over Time            │  │                          │
│  embedding_classifier  │  │  signal_pipeline.py   │  │  dashboard.py (8 tabs)   │
│  narrative_evolution   │  │  narrative_evolution  │  │  intervention_engine.py  │
│                        │  │                       │  │  rag_pipeline.py         │
│  ● Generate embeddings │  │  ● Weekly/Monthly     │  │  ● Risk signal viz       │
│    for forum posts     │  │    Binning            │  │  ● Concise summaries     │
│  ● Cluster posts into  │  │  ● Spike Detection    │  │  ● Supporting evidence   │
│    behavioral patterns │  │    (z-score on bins)  │  │  ● Ethical/auditable     │
│    (HDBSCAN/KMeans)   │  │  ● Recurring pattern  │  │    outputs               │
│  ● Identify evolving   │  │    detection          │  │  ● Intervention recs     │
│    narratives via 8    │  │  ● CDC cross-         │  │                          │
│    topic anchors       │  │    correlation        │  │  METRICS:                │
│  ● UMAP trajectory     │  │    (lead-lag r/p)     │  │  ● ROUGE-L / BERTScore   │
│                        │  │                       │  │    for RAG summaries     │
│  METRICS:              │  │  METRICS:             │  │  ● Faithfulness (Ragas)  │
│  ● Silhouette Score    │  │  ● MRR — how high is  │  │    for RAG groundedness  │
│    (cluster quality)   │  │    critical event in  │  │  ● Accuracy / F1-Score   │
│  ● NDCG — ranking of   │  │    alert priority?    │  │    for risk detection    │
│    high-risk trends    │  │  ● Detection Lag —    │  │                          │
│  ● Perplexity — LM fit │  │    time between spike │  │                          │
│    on domain slang     │  │    and system alert   │  │                          │
└────────────────────────┘  └───────────────────────┘  └──────────────────────────┘
                                    │
          ┌─────────────────────────▼──────────────────────────┐
          │         INNOVATION TESTING FRAMEWORK               │
          │                                                    │
          │  ● Metamorphic Testing: Replace drug term with     │
          │    NIDA slang → does risk flag persist?            │
          │  ● Property-Based Testing: Safety guards           │
          │    (no username displayed, PII-clean outputs)      │
          │  ● Human-in-the-Loop: Teammate acts as             │
          │    "Public Health Analyst", scores outputs 1–5     │
          └────────────────────────────────────────────────────┘
```

---

## Division of Work

### Member 1 — Tony Nguyen | Task 1: Embedding Trend Discovery & Behavioral Analysis

**Files:**
- ~~`src/classifiers/finetuned_classifier.py`~~ ✅ **DONE** — `finetuned_results.csv` (41,830 rows) produced
- ~~`src/eval/cluster_metrics.py`~~ ✅ **DONE** — `cluster_metrics.json` produced
- ~~`models/finetuned_bert/`~~ ✅ **DONE** — checkpoint saved
- ~~`requirements.txt`~~ ✅ **DONE** — `transformers`, `datasets`, `accelerate`, `torch`, `rouge-score`, `bert-score`, `ragas`, `beautifulsoup4`, `lxml` added
- ~~`src/classifiers/ensemble.py`~~ ✅ **DONE** — finetuned weight tables (`WEIGHTS_FULL_FT`, `WEIGHTS_NO_LLM_FT`), `FINETUNED_CSV` path, 4-return `load_classifier_outputs()`, `merge_classifiers()` finetuned branch, and voting logic all implemented

**Implementation Steps:**

1. **Fine-Tuned Classifier (4th method)** ✅ **DONE (Apr 1)**
   - Load `data/processed/posts_preprocessed.csv` + `data/processed/ensemble_results.csv` → use `final_risk_level` as pseudo-ground-truth labels
   - Fine-tune `distilbert-base-uncased` via HuggingFace `Trainer` on `processed_text` column (low=0, medium=1, high=2)
   - 80/20 stratified train/val split, 3 epochs, batch size 16
   - Save model checkpoint to `models/finetuned_bert/`
   - Run full inference → `data/processed/finetuned_results.csv` (columns: `post_id`, `risk_level`, `confidence`)
   - ✅ `ensemble.py` already updated — will auto-load `finetuned_results.csv` when the file exists

2. **Cluster Quality Metrics** (`src/eval/cluster_metrics.py`) ✅ **DONE (Apr 2)**
   - Compute **Silhouette Score** on HDBSCAN/KMeans cluster assignments from `data/processed/clustered_posts.csv`
   - Compute **NDCG** — treat "high-risk" posts as relevant; measure if the embedding similarity ranking surfaces high-risk clusters at top positions
   - Compute **Perplexity** — train a small unigram/bigram LM on processed posts; measure perplexity on the domain slang vocabulary (`SLANG_MAP` from `src/processing/preprocess_posts.py`)
   - Write all three scores to `data/processed/cluster_metrics.json`

3. **Wire finetuned into method_comparison.csv**  ✅ **DONE (Apr 2)**
   - After `finetuned_results.csv` is produced, update `src/eval/generate_comparison.py` to include a `"Finetuned"` row (columns: `finetuned_results.csv`, column `risk_level`) and re-run

**Outputs:** `finetuned_results.csv`, `cluster_metrics.json`, `models/finetuned_bert/`

---

### Member 2 — Daniel Evans | Task 2: Temporal Analysis Metrics + Evaluation Module

**Files:**
- ~~`src/eval/temporal_metrics.py`~~ ✅ **DONE** — `temporal_metrics.json` produced (MRR=0.1508 overall; opioid early-warning median −351 d)
- ~~`src/eval/evaluation_report.py`~~ ✅ **DONE** — `eval_test_set.csv` (2,000 posts), `eval_metrics.json`, 7 Plotly figures in `eval_figures/`
- ~~Expand `src/eval/generate_comparison.py`~~ ✅ **DONE** (wired finetuned; method_comparison.csv updated)

**Implementation Steps:**

1. **Temporal Metrics** (`src/eval/temporal_metrics.py`) ✅ **DONE (Apr 5)**
   - **MRR (Mean Reciprocal Rank):** For each known CDC overdose spike month in `data/processed/cdc_cleaned.csv`, check where the system-generated alert for that substance appears in the ranked alert list → MRR = mean(1/rank)
   - **Detection Lag:** For each spike event where both a CDC spike and a social signal spike exist, compute `lag = date_alert_issued - date_spike_started` in days. Report mean/median detection lag per substance.
   - Results: Overall MRR=0.1508 (opioid=0.2984, cocaine=0.1066, stimulant=0.0938); all 3 substances show early-warning median lag (opioid −351 d, cocaine −335 d, stimulant −335 d)
   - Written to `data/processed/temporal_metrics.json`

2. **Formal Evaluation Module** (`src/eval/evaluation_report.py`) ✅ **DONE (Apr 5)**
   - Built held-out test set: stratified 2,000-post sample from full corpus → `data/processed/eval_test_set.csv`
   - Computed per-method: Accuracy, macro-F1, per-class F1, ROC-AUC (one-vs-rest)
   - Per-substance F1 breakdown (opioid, stimulant, benzo, alcohol)
   - Results: Ensemble best overall (acc=0.404, macro-F1=0.304, AUC=0.506); Rule-based highest accuracy (0.495)
   - Built 7 Plotly figures: multi-class ROC curves (2×2), 4× confusion matrix heatmaps, per-substance F1 bar chart, summary bar chart
   - Saved to `data/processed/eval_figures/`

3. **Wire into dashboard** (`src/app/dashboard.py` Model Evaluation tab) ✅ **DONE (Apr 5)**
   - Added ROC curve chart, confusion matrix selector, per-substance F1 chart, per-method KPI cards
   - Added MRR and Detection Lag KPI cards to the Alerts tab
   - Added `_load_temporal_metrics()` and `_load_eval_metrics()` cached loaders

**Outputs:** `temporal_metrics.json`, `eval_test_set.csv`, `eval_metrics.json`, `eval_figures/` (7 HTML figures)

---

### Member 3 — Joel Vinas | Task 3: Dashboard Decision Support + Testing Framework

**Files:**
- ~~`src/agents/intervention_engine.py`~~ ✅ **DONE**
- ~~`src/eval/summary_metrics.py`~~ ✅ **DONE**
- ~~`src/tests/test_metamorphic.py`~~ ✅ **DONE**
- ~~`src/tests/test_safety_guards.py`~~ ✅ **DONE**
- ~~Add Tab 8 to `src/app/dashboard.py`~~ ✅ **DONE**

**Implementation Steps:**

1. **Intervention Recommendation Engine** (`src/agents/intervention_engine.py`)  ✅ **DONE (Apr 2)**
   - Read `warning_report.csv` + `correlations.json` + `cdc_cleaned.csv`
   - Rules:
     - Critical opioid/fentanyl alert → "Deploy naloxone in top-3 states by death rate"
     - Lead-lag N months → "Activate early warning protocol N months before projected spike"
     - Rising harm_reduction topic → "Increase harm reduction outreach messaging"
   - Severity tiers: `IMMEDIATE` / `MONITOR` / `INFORMATIONAL`
   - Write `data/processed/recommendations.json`
   - Add **Tab 8 "Recommendations"** to dashboard: sortable table with badges, substance filter, expandable rationale

2. **Summary Quality Metrics** (`src/eval/summary_metrics.py`)  ✅ **DONE (Apr 2)**
   - **ROUGE-L:** Compare RAG-generated summaries in `spike_summaries.json` against reference summaries (manually write 3–5 reference summaries for known spike events)
   - **BERTScore:** Semantic similarity between generated and reference summaries
   - **Faithfulness (Ragas):** Check that each claim in analyst summaries is grounded in retrieved evidence spans — flag any ungrounded sentences
   - Write scores to `data/processed/summary_metrics.json`

3. **Innovation Testing Framework**  ✅ **DONE (Apr 2)**
   - **Metamorphic Testing** (`src/tests/test_metamorphic.py`):
     - Take 50 high-risk posts, replace drug terms with NIDA slang equivalents from `preprocess_posts.py:SLANG_MAP`
     - Assert: risk label must remain "high" after substitution
     - Report pass/fail rate
   - **Property-Based Safety Guards** (`src/tests/test_safety_guards.py`):
     - For every post shown in the dashboard, assert no email, phone, ZIP, or name pattern appears (regex check)
     - For every analyst summary, assert it contains no personal pronouns tied to identifiers

**Outputs:** `recommendations.json`, `summary_metrics.json`, test pass/fail reports

---

### Member 4 — Tina Nguyen | Final Submission Report + Human-in-the-Loop Evaluation

**Files:**
- `report/nsf_nrt_challenge1_report.md` (create, export to PDF)
- `data/processed/hitl_scores.csv` (create — human annotation scores)

**Implementation Steps:**

1. **Human-in-the-Loop Annotation** — Act as "Public Health Analyst":
   - Take 10 randomly selected Analyst Report summaries from `spike_summaries.json`
   - Score each on: Clarity (1–5), Actionability (1–5), Faithfulness (1–5), Privacy-safe (Y/N)
   - Record in `data/processed/hitl_scores.csv`
   - Report mean scores in the report and dashboard (add a small HitL scorecard to Tab 6)

2. **Final 4-Page Report** (`report/nsf_nrt_challenge1_report.md`):
   - **Section 1: Introduction** — Problem, datasets, ethical constraints
   - **Section 2: System Architecture** — Pipeline diagram + 5-layer description
   - **Section 3: Methods & Results** — Classifier comparison table (Accuracy/F1/AUC), temporal findings (lead-lag N months, MRR, Detection Lag), cluster quality (Silhouette, NDCG, Perplexity)
   - **Section 4: Innovation Highlights** — Fine-tuned BERT, RAG + cross-encoder reranking, Metamorphic + Property-Based testing, HitL evaluation
   - **Section 5: Ethics & Privacy** — PII scrubbing, property-based safety guards, population-level-only outputs
   - **Section 6: Conclusion & Impact** — How the system enables proactive public health intervention

**Outputs:** `report/nsf_nrt_challenge1_report.md` (PDF), `hitl_scores.csv`

---

## Full Metrics Reference

| Metric | Task | Purpose | Owner |
|--------|------|---------|-------|
| **Silhouette Score** | Task 1 | Cluster distinctness & quality | M1 Tony |
| **NDCG** | Task 1 | High-risk trend ranking quality | M1 Tony |
| **Perplexity** | Task 1 | LM fit on domain slang vocab | M1 Tony |
| **MRR** | Task 2 | Alert priority ranking accuracy | M2 Daniel |
| **Detection Lag** | Task 2 | Time between spike and system alert | M2 Daniel |
| **Accuracy / F1** | Task 2/3 | Risk detection correctness | M2 Daniel |
| **ROC-AUC** | Task 2 | Per-method classifier quality | M2 Daniel |
| **ROUGE-L** | Task 3 | Summary text overlap quality | M3 Joel |
| **BERTScore** | Task 3 | Summary semantic similarity | M3 Joel |
| **Faithfulness (Ragas)** | Task 3 | RAG groundedness (no hallucination) | M3 Joel |
| **HitL Score** | Task 3 | Human-rated interpretability | M4 Tina |

---

## Timeline

**Today: April 1, 2026 (Day 2) | Target Completion: April 5, 2026 EOD**

| Day | Date | M1 Tony | M2 Daniel | M3 Joel | M4 Tina |
|-----|------|---------|-----------|--------|---------|
| ~~1~~ | ~~Mar 31~~ | ~~Set up HuggingFace env, install transformers, review cluster outputs~~ ✅ **DONE** — `requirements.txt` updated with all dependencies; `ensemble.py` fully extended with finetuned weight tables, CSV path, merge logic, and voting | ~~Build eval test set, define MRR + Detection Lag schema~~ | ~~Design intervention rules, review RAG outputs~~ | ~~Outline report, gather existing results~~ |
| ~~2~~ | ~~Apr 1~~ | ~~Create `src/classifiers/finetuned_classifier.py`~~ ✅ **DONE** — DistilBERT fine-tuned on 10k sample; `finetuned_results.csv` (41,830 rows) produced; Silhouette Score computed (0.0753) | Implement `temporal_metrics.py`, compute Accuracy/F1/ROC-AUC per method | Implement engine v1 (3 rule types), write ROUGE-L/BERTScore script | Write Sections 1 & 2 (intro + arch) |
| ~~3~~ | ~~Apr 2~~ | ~~Create `src/eval/cluster_metrics.py`~~ ✅ **DONE** — `cluster_metrics.py` implemented (Silhouette 0.0753, NDCG@100 0.1248, Perplexity 60,897); `cluster_metrics.json` produced; `generate_comparison.py` updated with Finetuned row → `method_comparison.csv` (4 methods: Ensemble best F1=0.413) | Build Plotly eval figures (ROC, confusion matrix, per-substance F1), wire into eval tab | Faithfulness (Ragas) check on RAG outputs, write Metamorphic test suite | Write Sections 3 & 4 (methods + innovation) |
| ~~4~~ | ~~Apr 3~~ | ~~Integration testing~~ ✅ **DONE** — Finetuned added to dashboard `_clf_files`; Cluster Quality Metrics panel (Silhouette / NDCG / Perplexity KPI cards + top-5 clusters expander) added to Model Evaluation tab; `method_comparison.csv` shows all 4 classifiers | Wire eval figures + temporal KPIs into dashboard, add MRR/Lag cards to Alerts tab | Build dashboard Tab 8 (table, filters, expandable rationale), add property-based safety guards | Write Sections 5 & 6 (ethics + conclusion), begin HitL annotation |
| ~~5~~ | ~~Apr 4~~ | ~~Buffer / assist teammates if needed; final smoke-test of full finetuned pipeline~~ ✅ **DONE** | ~~Integration testing — all metrics compute and display correctly~~ ✅ **DONE** — `temporal_metrics.py`, `evaluation_report.py` implemented; MRR/Lag KPIs + eval figures wired into dashboard | ~~Integration testing — Recommendations tab + tests passing, `summary_metrics.json` populated~~ ✅ **DONE** | Complete HitL annotation, wire HitL scorecard into Tab 6, draft PDF |
| 6 | **Apr 5** | **FINAL REVIEW DAY** — All members run full pipeline, validate all metrics, finalize PDF report. All tests (metamorphic + safety guards) pass. Submission package assembled by EOD. | | | |
| 7 | **Apr 6** | **SUBMIT by 12:00 PM NOON** | | | |

---

## Critical Files

| File | Owner | Status |
|------|-------|--------|
| `scripts/fetch_erowid.py` | M1 Tony | ✅ Done — git clone Erowid mirror; idempotent |
| `scripts/process_erowid_lsa.py` | M1 Tony | ✅ Done — `erowid_substance_profiles.json` + similarity JSON (19 substances) |
| `src/processing/process_erowid.py` | M1 Tony | ✅ Done — BS4 HTML parse + PII scrub + risk label → `erowid_posts.csv` |
| `src/processing/process_erowid_lsa.py` | M1 Tony | ✅ Done — LSA variant → `erowid_posts.csv` (356 rows) |
| `data/processed/erowid_substance_profiles.json` | M1 Tony | ✅ Done — NMF topic profiles |
| `data/processed/erowid_substance_similarity.json` | M1 Tony | ✅ Done — 196 substance pairs |
| `src/classifiers/finetuned_classifier.py` | M1 Tony | ✅ Done — `finetuned_results.csv` (41,830 rows) produced |
| `models/finetuned_bert/` | M1 Tony | ✅ Done — checkpoint saved |
| `src/classifiers/ensemble.py` | M1 Tony | ✅ Done — finetuned 4th method fully integrated |
| `src/eval/cluster_metrics.py` | M1 Tony | ✅ Done — `cluster_metrics.json` produced (Silhouette 0.0753, NDCG@100 0.1248) |
| `src/eval/temporal_metrics.py` | M2 Daniel | ✅ Done — `temporal_metrics.json` produced (MRR 0.1508, early-warning lag −335 to −351 d) |
| `src/eval/evaluation_report.py` | M2 Daniel | ✅ Done — `eval_metrics.json` + 7 Plotly figures in `eval_figures/` |
| `src/eval/generate_comparison.py` | M2 Daniel | ✅ Done — finetuned row wired; `method_comparison.csv` updated |
| `data/processed/eval_test_set.csv` | M2 Daniel | ✅ Done — 2,000 posts, stratified by GT risk level |
| `data/processed/eval_figures/` | M2 Daniel | ✅ Done — 7 HTML figures (ROC, CM×4, per-substance F1, summary) |
| `src/agents/intervention_engine.py` | M3 Joel | ✅ Done |
| `src/eval/summary_metrics.py` | M3 Joel | ✅ Done — computed ROUGE-L, BERTScore, and Faithfulness |
| `src/tests/test_metamorphic.py` | M3 Joel | ✅ Done — metamorphic substitution tests passing |
| `src/tests/test_safety_guards.py` | M3 Joel | ✅ Done — PII checks passing |
| `data/processed/recommendations.json` | M3 Joel | ✅ Done |
| `src/agents/signal_pipeline.py` | M1 Tony | ✅ Done — `load_erowid_posts()` + `merge_post_sources()` added; CDC + Erowid + drug-review signal merged |
| `src/app/dashboard.py` | M2 + M3 + M4 | ✅ Partially done — source filter (Drug Reviews / Erowid / Both), `_load_post_sources()`, data-source sidebar panel added; eval tab fully wired (ROC, CM, per-substance F1, KPI cards, MRR/Lag section); HitL scorecard pending (M4) |
| `data/processed/hitl_scores.csv` | M4 Tina | Create |
| `report/nsf_nrt_challenge1_report.md` | M4 Tina | Create |
| `requirements.txt` | M1 Tony | ✅ Done — `transformers`, `datasets`, `accelerate`, `torch`, `rouge-score`, `bert-score`, `ragas` added |

---

## Erowid LSA Integration (Signal Enrichment — Additive)

> **PI-approved:** Aggregate signal discovery only. No individual reports stored or surfaced. All outputs are population-level. Integrated with CDC/NIDA per PI guidance.

### What It Adds

| Layer | Enhancement |
|-------|-------------|
| **Risk Detection** | Substance weight boost (0.05–0.15) for substances whose NMF dominant terms overlap with harm vocabulary (overdose, withdrawal, needle, seizure, ...) |
| **Preprocessing** | Slang lexicon expansion with substance-specific terms surfaced by NMF (k-hole, rolling, shrooms, ...) |
| **Temporal Analysis** | Spillover spike detection — when one substance spikes, check if NMF-similar substances co-spike within ±35 days |
| **Explainability (RAG)** | Aggregate NMF topic terms injected into Gemini context before analyst summaries; `erowid_nmf_context` field in `spike_summaries.json` |

### New Files

| File | Role | Status |
|------|------|--------|
| `scripts/fetch_erowid.py` | Download erowid-scrape.py + run scraper (urllib direct download — avoids Windows git path issues) | Done |
| `scripts/process_erowid_lsa.py` | TF-IDF + NMF (8 topics) on scraped reports -> `erowid_substance_profiles.json` + `erowid_substance_similarity.json` | Done |
| `src/processing/process_erowid_lsa.py` | Parse Erowid HTML reports -> `erowid_posts.csv` (schema-compatible with `posts_classified.csv`) | Done |
| `data/raw/erowid-lsa-repo/erowid-scrape.py` | Python 3 port of original scraper with Windows-safe filenames, rate limiting, correct request params | Done |

### Modified Files

| File | Change | Status |
|------|--------|--------|
| `src/classifiers/rule_based_classifier.py` | `load_erowid_substance_boost()` + `_EROWID_BOOSTS` applied in `score_substance_mentions()` | Done |
| `src/processing/preprocess_posts.py` | `load_erowid_slang_extensions()` appended to `SLANG_LEXICON` before pattern compilation | Done |
| `src/agents/signal_pipeline.py` | `load_erowid_similarity_graph()` + `detect_spillover_spikes()` -> `erowid_spillover.csv` | Done |
| `src/agents/rag_pipeline.py` | `_load_erowid_profile_for_substance()` prepended to `flagged_posts`; `erowid_nmf_context` in output | Done |

### Artifacts

| File | Contents |
|------|----------|
| `data/processed/erowid_substance_profiles.json` | NMF topic profiles per substance (19 substances, 212 reports, 8 topics each) |
| `data/processed/erowid_substance_similarity.json` | Pairwise cosine similarity between substance NMF vectors (196 pairs) |
| `data/processed/erowid_spillover.csv` | Co-spiking substance pairs flagged by spillover detection |
| `data/raw/erowid_posts.csv` | 356 classified Erowid experience reports (low=190, medium=83, high=83) |

### Run Order

```powershell
python scripts/fetch_erowid.py                  # download + scrape (network-bound, 10-30 min)
python src/processing/process_erowid_lsa.py     # classify reports -> erowid_posts.csv
python scripts/process_erowid_lsa.py            # NMF analysis -> substance profiles + similarity JSON
# then run existing pipeline as normal
```

### Graceful Degradation

All 4 integration points check for file existence at import and return empty dicts / `None` if either JSON is missing. The full pipeline produces identical results to the pre-integration baseline when Erowid files are absent.

---

## Existing Code to Reuse

| Resource | Use |
|----------|-----|
| `src/classifiers/ensemble.py` | Extend with finetuned weight |
| `src/eval/generate_comparison.py` | Existing metric computation — expand, don't replace |
| `src/app/dashboard.py` | Follow existing `st.tabs()` pattern for new tabs |
| `data/processed/ensemble_results.csv` | Pseudo-labels for fine-tuning training data |
| `data/processed/narrative/warning_report.csv` | Alert inputs for intervention engine |
| `data/processed/correlations.json` | Lead-lag rationale for recommendations |
| `src/processing/preprocess_posts.py:SLANG_MAP` | Slang vocabulary for metamorphic tests |

---

## Verification Checklist (April 5)

**M1 Tony**
- [x] `requirements.txt` updated with `transformers`, `datasets`, `accelerate`, `torch`, `rouge-score`, `bert-score`, `ragas`, `beautifulsoup4>=4.12`, `lxml>=5.0`
- [x] `src/classifiers/ensemble.py` — finetuned 4th method weight tables, CSV path, merge, voting all implemented
- [x] `python src/classifiers/finetuned_classifier.py` → `finetuned_results.csv` (41,830 rows: high=2.9%, med=58.3%, low=38.8%)
- [x] `models/finetuned_bert/` saved checkpoint exists
- [x] `python src/eval/cluster_metrics.py` → `cluster_metrics.json` with Silhouette=0.0753, NDCG@100=0.1248, Perplexity=60,897
- [x] `python src/eval/generate_comparison.py` → `method_comparison.csv` has 4 methods; Ensemble best (Acc=0.518, F1=0.413)
- [x] Finetuned classifier visible in dashboard Model Evaluation tab (`_clf_files` updated)
- [x] Cluster Quality Metrics panel added to dashboard (Silhouette / NDCG / Perplexity KPI cards)
- [x] `src/processing/process_erowid.py` → `erowid_posts.csv` produced (BS4 parse, PII scrub, risk labels)
- [x] `src/agents/signal_pipeline.py` → `load_erowid_posts()` + `merge_post_sources()` integrated; Erowid + Kaggle posts merged for signal
- [x] `src/app/dashboard.py` → source filter radio (Drug Reviews / Erowid Narratives / Both) + data-source counts in sidebar

**M2 Daniel**
- [ ] `python src/eval/temporal_metrics.py` → `temporal_metrics.json` with MRR, Detection Lag
- [ ] `python src/eval/evaluation_report.py` → ROC + confusion matrix figures in `eval_figures/`

**M3 Joel**
- [x] `python src/agents/intervention_engine.py` → `recommendations.json` produced
- [x] `python src/eval/summary_metrics.py` → `summary_metrics.json` with ROUGE-L, BERTScore, Faithfulness
- [x] `python src/tests/test_metamorphic.py` → ≥80% pass rate
- [x] `python src/tests/test_safety_guards.py` → 100% pass rate (zero PII in outputs)

**Integration**
- [ ] `streamlit run src/app/dashboard.py` → 8 tabs visible; all metrics display; Tab 8 renders
- [ ] `data/processed/hitl_scores.csv` has 10 annotated summaries
- [ ] `report/nsf_nrt_challenge1_report.md` ≤ 4 pages, all sections complete
