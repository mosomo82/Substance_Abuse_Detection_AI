# Individual Contribution Statement
## NSF NRT Research-A-Thon 2026 — Challenge 1, Track B

**Name:** Tina Nguyen
**GitHub:** tnguyen-umkc
**Date:** April 6, 2026

---

## 1. Role

**Technical Documentation & Human-in-the-Loop (HITL) Specialist**

Tina served as the primary technical writer and human-validation specialist for the project. Her responsibility was to translate technical achievements into the project's final submission materials. She authored the 4-page research report and coordinated the assembly of the team's individual contribution statements. Additionally, she designed and executed the Human-in-the-Loop (HITL) audit to provide a qualitative validation of the machine learning metrics.

---

## 2. Personal Contribution

### 2.1 Final Report Writing (`report/nsf_nrt_challenge1_report.md`)
Responsible for the project's 4-page formal report.

- Synthesized technical findings from all members into a cohesive **Problem-Solution-Evaluation** narrative.
- Authored core sections: **Problem Statement**, **Experimental Design**, and **Ethical Considerations**.
- Managed the **LaTeX/PDF pipeline**, ensuring the final document met all submission formatting requirements.

---

### 2.2 Human-in-the-Loop (HITL) Evaluation
Designed and executed the manual auditing protocol for validation.

- Conducted a **10-post stratified audit** to verify the ensemble's high-risk classifications.
- Formulated the **HITL Scoring Rubric** (Accuracy, Actionability, Tone) used to benchmark performance.
- Documented the **6/10 approval rate** and the identified failure modes for the report.

---

### 2.3 Team Contribution Coordination
Led the assembly of the project's individual contribution documentation.

- **Consolidation**: Coordinated with the team to gather evidence of work and technical reflections.
- **Editing & Consistency**: Managed all four `_contribution.md` files to ensure consistent formatting and reporting.
- **Metadata Management**: Ensured all GitHub artifacts were correctly named and tagged for final submission.

---

### 2.4 Project Branding & Poster Collaboration
Contributed to the visual communication of the project's results.

- **Branding & Logo**: Created the project's visual identity and color palette for the report.
- **Poster Narrative**: Collaborated on the scientific poster's text-heavy sections to ensure the value proposition was clear.

---

## 3. Division of Work

| Task | Tony | Daniel | Joel | Tina |
| :--- | :---: | :---: | :---: | :---: |
| Data preprocessing | Reviewed | — | Primary | — |
| Rule-based classifier | — | — | Primary | Reviewed |
| **Embedding classifier** | **Primary** | — | — | — |
| **LLM classifier** | **Shared** | — | Shared | — |
| **Fine-tuned BERT** | **Primary** | — | — | Reviewed |
| **Ensemble fusion** | **Primary** | — | Reviewed | — |
| Temporal analysis | — | Primary | — | Reviewed |
| **Cluster quality metrics** | **Primary** | Reviewed | — | — |
| Evaluation report | Reviewed | Primary | — | — |
| RAG pipeline | — | Reviewed | Primary | — |
| Dashboard | — | Reviewed | Primary | — |
| Intervention engine | — | — | Primary | Reviewed |
| **HITL evaluation** | — | — | — | **Primary** |
| Metamorphic testing | — | — | Primary | — |
| **Final report writing** | — | — | — | **Primary** |
| **Contribution Coordination** | Reviewed | — | — | **Primary** |
| Demo video creation | — | Primary | — | — |
| Poster Design | Primary | Reviewed | — | Reviewed |

Tina contributed to approximately **10%** of the codebase and served as the project's primary reporting specialist.

---

## 4. Evidence of Works

| Artifact | Location | Description |
| :--- | :--- | :--- |
| `nsf_nrt_challenge1_report.md` | `report/` | Final 4-page research report |
| `tina_nguyen_contribution.md` | `report/CONTRIBUTION/` | Consolidated team contribution files |
| `HITL Audit Results` | `report/` | Documentation of the 60% approval rating |
| `nsf_nrt_challenge1_report.pdf` | `report/` | Compiled PDF submission artifact |

---

## 5. Technical Reflection

### What worked well
The **Report-to-Code mapping** was a highlight. By having a dedicated documentation focus, we were able to ensure that the code headers and the report sections used identical terminology. The **HITL results** also added a critical dimension of validity.

### Challenges and lessons learned
The **LaTeX transition** for the final PDF was the most complex non-code task. Ensuring that all tables and diagrams fit within the strict 4-page limit required extensive troubleshooting.
