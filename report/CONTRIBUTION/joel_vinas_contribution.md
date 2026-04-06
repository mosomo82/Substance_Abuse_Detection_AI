# Individual Contribution Statement
## NSF NRT Research-A-Thon 2026 — Challenge 1, Track B

**Name:** Joel Vinas
**GitHub:** jvinas
**Date:** April 6, 2026

---

## 1. Role

**Dashboard Decision Support & Testing Framework Specialist**

Joel served as the specialist for Dashboard Decision Support and the Innovation Testing Framework (Task 3). His primary responsibilities included developing an intervention recommendation engine, establishing quality metrics for the RAG-generated text, and building a robust testing suite containing metamorphic and property-based safety tests. He also integrated these systems into the final user-facing dashboard.

---

## 2. Personal Contribution

### 2.1 Intervention Recommendation Engine (`src/agents/intervention_engine.py`)
Responsible for building the actionable rules engine.

- Designed an engine that reads `warning_report.csv`, `correlations.json`, and `cdc_cleaned.csv` to generate systemic recommendations.
- Implemented specific rules for deployments, such as early warning protocol triggers based on lead-lag correlations and naloxone deployment strategies for high-risk opioid alerts.
- Configured severity tiers (`IMMEDIATE`, `MONITOR`, `INFORMATIONAL`) and generated `recommendations.json`.

---

### 2.2 Summary Quality Metrics (`src/eval/summary_metrics.py`)
Developed the automated evaluation pipeline for the Analyst Reports.

- Implemented **ROUGE-L** and **BERTScore** to compare RAG-generated summaries against manually written references.
- Integrated **Faithfulness (Ragas)** to ensure that generated analyst claims are grounded in retrieved evidence spans, effectively testing for hallucinations.
- Logged all text generation quality metrics to `data/processed/summary_metrics.json`.

---

### 2.3 Innovation Testing Framework (`src/tests/test_metamorphic.py`, `src/tests/test_safety_guards.py`)
Led the system's robustness, safety, and privacy evaluations.

- **Metamorphic Testing**: Built tests replacing drug terms with NIDA slang vocabulary in high-risk posts, asserting that risk classification labels remain durable.
- **Safety Guards**: Implemented property-based regex tests to guarantee that no PII (email, phone, ZIP, or name layout) slips into the dashboard, and that analyst summaries exclude identifiers or personal pronouns.

---

### 2.4 Dashboard Integration (`src/app/dashboard.py`)
Wired the decision support engine to the Streamlit UI.

- Created **Tab 8 ("Recommendations")** in the main user dashboard.
- Built a sortable interactive table with badges, substance filters, and expandable rationales using data parsed from `recommendations.json`.

---

## 3. Division of Work

| Task | Tony | Daniel | Joel | Tina |
| :--- | :---: | :---: | :---: | :---: |
| Data preprocessing | Reviewed | — | **Primary** | — |
| Rule-based classifier | — | — | **Primary** | Reviewed |
| Embedding classifier | Primary | — | — | — |
| LLM classifier | Shared | — | **Shared** | — |
| Fine-tuned BERT | Primary | — | — | Reviewed |
| Ensemble fusion | Primary | — | Reviewed | — |
| Temporal analysis | — | Primary | — | Reviewed |
| Cluster quality metrics | Primary | Reviewed | — | — |
| Evaluation report | Reviewed | Primary | — | — |
| **RAG pipeline** | — | Reviewed | **Primary** | — |
| **Dashboard** | — | Reviewed | **Primary** | — |
| **Intervention engine** | — | — | **Primary** | Reviewed |
| HITL evaluation | — | — | — | Primary |
| **Metamorphic testing** | — | — | **Primary** | — |
| Final report writing | — | — | — | Primary |
| Contribution Coordination | Reviewed | — | — | Primary |
| Demo video creation | — | Primary | — | — |
| Poster Design | Primary | Reviewed | — | Reviewed |

Joel contributed as a Primary implementer across the data processing, UI, systems testing, and decision support parts of the project.

---

## 4. Evidence of Works

| Artifact | Location | Description |
| :--- | :--- | :--- |
| `intervention_engine.py` | `src/agents/` | Generates public health responses to risk signals |
| `summary_metrics.py` | `src/eval/` | Computes ROUGE-L, BERTScore, and Faithfulness |
| `test_metamorphic.py` | `src/tests/` | Sub-tests slang replacements for robustness |
| `test_safety_guards.py` | `src/tests/` | Asserts privacy checks prior to display |
| `recommendations.json` | `data/processed/` | Structured output from the intervention rules |
| `summary_metrics.json` | `data/processed/` | RAG summary quality scoring |
| `dashboard.py` (Tab 8) | `src/app/` | UI implementation of Recommendation engines |

---

## 5. Technical Reflection

### What worked well
The **Metamorphic Testing framework** proved extremely valuable. By systematically substituting standard terms with street aliases, we mathematically validated that the model's performance doesn't solely rely on obvious features. The **ROUGE-L/BERTScore checks** also gave us confidence that generated RAG summaries remained faithful to actual post contents.

### Challenges and lessons learned
Constructing the **Intervention Recommendation Rules** required a high-level orchestration of different output pieces (correlations, CDC data, warning reports). Managing Streamlit tab states and connecting raw JSON pipeline outputs to visually pleasing `st.data_editor` / badges took significant layout consideration.
