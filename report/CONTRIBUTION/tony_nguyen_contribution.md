# Individual Contribution Statement
## NSF NRT Research-A-Thon 2026 — Challenge 1, Track B

**Name:** Tony Nguyen
**GitHub:** mosomo82
**Date:** April 6, 2026

---

## 1. Role

**Model Builder — Classification Pipeline, Embeddings & Ensemble**

Tony served as the primary machine learning engineer for the project, owning the entire classification pipeline from feature representation through multi-method fusion. His responsibility spanned four of the six technical layers of the system: the embedding-based classifier, the fine-tuned BERT classifier, the ensemble fusion module, and the cluster quality evaluation metrics. He was also responsible for maintaining the internal consistency of the ML outputs (i.e., ensuring all four classifiers produced compatible schema) so that downstream modules (temporal analysis, RAG, dashboard) could consume them reliably.

---

## 2. Personal Contribution

### 2.1 Embedding-Based Classifier (`src/classifiers/embedding_classifier.py`)
Designed and implemented the Layer 2 detection module from scratch.

- Selected **`all-MiniLM-L6-v2`** (Sentence-BERT) as the encoder after benchmarking against `paraphrase-mpnet-base-v2` on the drug-review seed bank — chosen for its 6× speed advantage at equivalent F1 on the substance detection task.
- Built a **FAISS-backed seed bank** (`seed_bank.faiss`, `seed_bank_labels.npy`) of manually curated high-risk post embeddings. Cosine similarity between a new post's embedding and the seed bank vectors produces per-class risk scores.
- Implemented **K-Means clustering (k=8) on UMAP-reduced embeddings** for behavioral sub-population discovery, producing `clustered_posts.csv` and `cluster_info.json`. Chose k=8 after elbow-method testing on silhouette scores across k=4–16.
- Designed the `retrieve_similar_posts()` API used by the RAG explainability layer (Task 3) to fetch top-k evidence posts for detected temporal spikes.

**Outputs produced:** `embedding_results.csv`, `clustered_posts.csv`, `cluster_info.json`

---

### 2.2 Fine-Tuned BERT Classifier (`src/classifiers/finetuned_classifier.py` + `notebooks/colab_finetuned_classifier.ipynb`)
Designed and executed the DistilBERT fine-tuning pipeline.

- Developed a **pseudo-label pipeline**: extracted `final_risk_level` labels from `ensemble_results.csv` to create 41,934 labeled training examples, bypassing the need for expensive manual annotation.
- Fine-tuned `distilbert-base-uncased` with a 3-class classification head (low / medium / high) using HuggingFace `Trainer` with early stopping (patience=2), weight decay, and warmup scheduling. Training was orchestrated via a Colab TPU notebook to handle the compute requirements.
- Implemented label-stratified 80/10/10 train/val/test splitting and per-epoch F1 tracking with a custom `compute_metrics` callback.
- Saved the full checkpoint to `models/finetuned_bert/` for reproducibility.

**Outputs produced:** `models/finetuned_bert/` (checkpoint), `data/processed/finetuned_results.csv`

---

### 2.3 Ensemble Fusion Module (`src/classifiers/ensemble.py`)
Designed the weighted voting strategy that fuses all four classifiers.

- Implemented **risk-to-integer encoding** (low=0, medium=1, high=2) with normalized weighted averaging, then decoded back to 3-class labels using calibrated thresholds (cut-points at 0.50 and 1.50).
- Designed adaptive weight re-normalization: when LLM predictions are absent (due to hybrid routing), weights shift from `{Rule: 0.25, Embed: 0.35, LLM: 0.40}` to `{Rule: 0.35, Embed: 0.65}` automatically, ensuring no silent degradation when the LLM API is rate-limited.
- Extended the ensemble to support the fine-tuned BERT as a 4th signal (`{Rule: 0.25, Embed: 0.30, LLM: 0.35, FT: 0.10}`), with weight tuning validated against the held-out test set.
- Produced `method_agreement` and `sources_used` columns in the output for downstream HITL auditing.

**Output produced:** `data/processed/ensemble_results.csv`

---

### 2.4 Cluster Quality Metrics (`src/eval/cluster_metrics.py`)
Solely authored the Task 1 evaluation module.

- Implemented **cosine-space Silhouette Score** on a stratified 8,000-post sample (full 40K × 384 matrix too large for memory-efficient computation on CPU).
- Implemented **NDCG@K** (K = 50, 100, 500) at both post level (ranking individual high-risk posts) and cluster level (ranking clusters by aggregate risk similarity to high-risk centroid). Cluster-level NDCG = 0.960 was the headline result used in the report.
- Implemented **bigram LM perplexity** on the domain slang vocabulary as a proxy for how well the embedding space models the drug-discourse distribution.

**Output produced:** `data/processed/cluster_metrics.json`

---

### 2.5 LLM Classifier Contribution (`src/classifiers/llm_classifier.py`)
Collaborated on the Gemini Flash integration.

- Co-designed the **hybrid routing logic**: posts with combined rule+embedding score ≥ 0.75 or ≤ 0.10 bypass the LLM API call (cost/latency optimization); only ambiguous posts in the 0.10–0.74 range are routed to Gemini Flash.
- Wrote the few-shot prompt template and JSON output schema for structured evidence-span extraction.

---

## 3. Division of Work

| Task | Tony | Daniel | Joel | Tina |
| :--- | :---: | :---: | :---: | :---: |
| Data preprocessing (`preprocess_posts.py`) | Reviewed | — | Lead | — |
| Rule-based classifier | — | — | Lead | Reviewed |
| **Embedding classifier** | **Lead** | — | — | — |
| **LLM classifier (hybrid routing)** | **Co-Lead** | — | Co-Lead | — |
| **Fine-tuned BERT** | **Lead** | — | — | Reviewed |
| **Ensemble fusion** | **Lead** | — | Reviewed | — |
| Temporal analysis / signal pipeline | — | Lead | — | Reviewed |
| **Cluster quality metrics** | **Lead** | Reviewed | — | — |
| Evaluation report / ROC figures | Reviewed | Lead | — | — |
| RAG pipeline | — | Reviewed | Lead | — |
| Dashboard (`dashboard.py`) | — | Reviewed | Lead | — |
| Intervention engine | — | — | Lead | Reviewed |
| HITL evaluation | — | — | — | Lead |
| Metamorphic / safety testing | — | — | Lead | — |
| Final report writing | — | — | — | Lead |
| Demo video creation | — | **Lead** | — | — |

Tony owned approximately **40%** of the total codebase by file count and was the primary technical decision-maker for all ML modeling choices.

---

## 4. Evidence of Works

| Artifact | Location | Description |
| :--- | :--- | :--- |
| `embedding_classifier.py` | `src/classifiers/` | Full embedding pipeline, FAISS seed bank, K-Means clustering |
| `finetuned_classifier.py` | `src/classifiers/` | DistilBERT fine-tuning with pseudo-labels and early stopping |
| `colab_finetuned_classifier.ipynb` | `notebooks/` | TPU Colab notebook for fine-tuning (training logs embedded) |
| `ensemble.py` | `src/classifiers/` | Weighted ensemble fusion with adaptive re-normalization |
| `cluster_metrics.py` | `src/eval/` | Silhouette, NDCG@K, and perplexity evaluation |
| `models/finetuned_bert/` | `models/` | Saved DistilBERT checkpoint (tokenizer + weights) |
| `embedding_results.csv` | `data/processed/` | Per-post embeddings, risk scores, cosine similarities |
| `clustered_posts.csv` | `data/processed/` | Post-to-cluster assignments (k=8) |
| `cluster_info.json` | `data/processed/` | Cluster size, dominant substance, representative examples |
| `cluster_metrics.json` | `data/processed/` | Silhouette=0.082, cluster-NDCG=0.960, post-NDCG@50=0.196 |
| `finetuned_results.csv` | `data/processed/` | Fine-tuned BERT predictions (risk_level, confidence) per post |
| `ensemble_results.csv` | `data/processed/` | Final fused predictions with method_agreement flags |

---

## 5. Primary Tools Used

| Category | Tool / Library | Version / Notes |
| :--- | :--- | :--- |
| **Language model** | `sentence-transformers` (`all-MiniLM-L6-v2`) | 384-dim dense embeddings |
| **Fine-tuning** | HuggingFace `transformers` + `datasets` | `distilbert-base-uncased`, 3-class head |
| **Training compute** | Google Colab TPU (T4 GPU) | Via `colab_finetuned_classifier.ipynb` |
| **Vector search** | `faiss-cpu` | FAISS IndexFlatIP for seed bank retrieval |
| **Clustering** | `scikit-learn` `KMeans` | k=8, cosine affinity via UMAP reduction |
| **Dimensionality reduction** | `umap-learn` | 2D + 50D projections for clustering |
| **Metrics** | `scikit-learn` (`silhouette_score`) + custom NDCG | See `cluster_metrics.py` |
| **LLM API** | Google Gemini Flash (`google-generativeai`) | Structured JSON output mode |
| **Data wrangling** | `pandas`, `numpy` | |
| **Deep learning framework** | `PyTorch` | Backend for HuggingFace `Trainer` |
| **Version control** | Git / GitHub (`mosomo82`) | |

---

## 6. Technical Reflection

### What worked well
The **pseudo-label fine-tuning approach** was the project's most technically interesting decision. Rather than spending weeks on manual annotation, the ensemble output was used as noisy-but-cheap supervision for DistilBERT. This allowed a contextual language model to be trained in hours, and the resulting model captured syntactic dependency patterns that escaped both the regex lexicon and the cosine-similarity scorer — particularly in posts that mentioned substances in a clinical or harm-reduction context without using high-risk slang.

The **adaptive ensemble re-normalization** was quietly important for production reliability. Early pipeline runs frequently hit Gemini Flash rate limits, which caused silent `None` entries in the LLM column. Unlike a naive average that would collapse to NaN, the adaptive weighting automatically redistributed mass to the remaining methods, keeping recall stable during API outages.

### Challenges and lessons learned
The **pseudo-label ceiling** is the sharpest limitation of the fine-tuned model. Because the training labels are derived from the ensemble, the fine-tuned BERT cannot exceed the ensemble's error floor — any systematic bias in the ensemble (sarcasm FP, novel-slang FN) is baked into the pseudo-labels and reproduced by the fine-tuned model. The right fix is a small high-quality human-annotated set (500–1,000 posts) to seed a semi-supervised approach.

The **class imbalance** (9.75% high-risk in test set) was not adequately addressed. In retrospect, implementing class-weighted loss (`weight=[1, 1, 10]` for `CrossEntropyLoss`) or oversampling the high-risk stratum with SMOTE-equivalent text paraphrasing via the LLM would have meaningfully improved high-risk recall, which is the clinically critical metric.

The **UMAP + K-Means pipeline** is sensitive to UMAP's `random_state` and `n_neighbors` hyperparameters, and small Silhouette scores (0.082) reflect the genuine semantic overlap of drug-discourse clusters rather than a modeling failure. Future work should explore HDBSCAN, which does not require pre-specifying k and handles variable-density clusters more naturally.

### Key metric summary
| Metric | Value | Context |
| :--- | :---: | :--- |
| Ensemble macro-F1 | 0.304 | Best of all four methods |
| Ensemble ROC-AUC | 0.506 | Best of all four methods |
| Benzo F1 (ensemble) | 0.366 | Best per-substance performance |
| Cluster-level NDCG | 0.960 | Cluster risk ranking quality |
| Silhouette (cosine) | 0.082 | Reflects semantic overlap |
| Opioid lead-lag r | 0.471 | At lag −3 months vs CDC data |
