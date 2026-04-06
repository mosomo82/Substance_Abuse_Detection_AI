# Lab 10 Demo Video — Scene Script

**Assignment:** Lab 10 / NSF NRT Research-A-Thon Challenge 1 (AI Challenge)
**Max length:** 3 minutes
**Topic:** AI for Substance Abuse Risk Detection from Social Signals

---

## Scene 1 — The Problem (~20s)

**Screen:** `C:\Users\danie\Documents\CS5542\lab10-scratch\title-card-slide.svg`

> "Substance abuse is a public health crisis — overdose deaths in the U.S. have surged for decades. But the warning signs often show up first in social media: posts about drug use, emotional distress, relapse. The goal of this project is to build an AI system that reads those posts, identifies rising risk, and gives public health analysts clear, explainable insights — before a crisis peaks."

---

## Scene 2 — Dataset & Architecture (~35s)

**Screen:** Architecture SVG (`data/figures/substance_abuse_ai_pipeline_track_b_architecture.svg`), highlight each layer as you mention it

> "We work with about 42,000 labeled drug-review posts from Kaggle, supplemented with community experience reports from Erowid, and validated against CDC overdose data and national disorder prevalence rates. Before any analysis, the text goes through preprocessing: cleaning, removing personal information, and normalizing drug slang — including street terms pulled from Erowid to keep up with evolving terminology. From there, three tasks. Task 1 is risk detection: four different methods running in parallel. Task 2 is temporal analysis: tracking how signals change over time and correlating them with real-world overdose data. Task 3 is explainability — so every insight an analyst sees is backed by actual source posts, not just a number."

---

## Scene 3 — Demo: Drift & Topics (~30s)

**Screen:** Dashboard — Drift tab, then Topics tab (can be one continuous take, switch tabs)

> "Here's the live dashboard. The Drift tab shows two signals over time: how fast the conversation is changing week to week, and what percentage of posts are flagged as high risk. The stars mark weeks where the shift was statistically significant — something meaningfully different was being talked about. Switching to the Topics tab: this stacked area chart shows eight topic clusters over time — things like opioid use, emotional distress, and polydrug mentions. You can watch these themes rise and fall, and spot when multiple topics spike together, which is an early sign of a broader crisis wave."

---

## Scene 4 — Demo: UMAP & Alerts (~25s)

**Screen:** Dashboard — UMAP Trajectory tab, then Alerts tab

> "This is a map of every week of posts compressed into 2D. Weeks with semantically-similar conversations land close together. The color shows risk level — darker red means more high-risk posts that week. What this chart shows is that the high-risk weeks aren't scattered randomly: they cluster toward the right side of the map. The left side is where the conversation was stable and lower-risk; the right side is where it shifted into higher-risk patterns. That separation is meaningful — it means there's a recognizable signal to how people talk during crisis periods, and the system can detect when the conversation heading in that direction. Then the Alerts tab surfaces those moments explicitly — each row flags a topic that's trending up, with the magnitude of the increase."

---

## Scene 5 — Demo: Model Evaluation (~25s)

**Screen:** Dashboard — Model Evaluation tab (metrics table + bar chart)

> "The Model Evaluation tab shows how differently each method behaves. The bar chart breaks down what each one actually does with a post — whether it calls it high, medium, or low risk. The embedding method is aggressive: it flags a third of all posts as high risk. The fine-tuned BERT is the opposite — barely 2% high risk, very conservative. The ensemble sits in the middle: it spreads the signal across high and medium, which is exactly what you want for a decision-support tool — not crying wolf on everything, but not missing real signals either."

---

## Scene 6 — Key Findings (~25s)

**Screen:** `C:\Users\danie\Documents\CS5542\lab10-scratch\cdc-correlation-slide.svg`

> "The biggest finding is this: opioid-related posts on the platform started spiking about three weeks before CDC-reported overdose deaths went up — with a correlation of 0.47 and a p-value of 0.006. That's statistically significant. It means the social signal is a leading indicator, not just a reflection of what already happened. We also found a spillover pattern — when opioid language starts showing up in posts that are normally about stimulants, multi-substance overdose spikes tend to follow. The system flags this automatically."

---

## Scene 7 — Team Summary (~15s)

**Screen:** `C:\Users\danie\Documents\CS5542\lab10-scratch\team-summary-slide.svg`

> "This system was built by Tony Nguyen, Daniel Evans, Joel Vinas, and Tina Nguyen. Tony built the core classifiers — BERT fine-tuning, embeddings, and RAG. Daniel handled the project presentation and demo. Joel built the dashboard, intervention engine, and testing framework. Tina led the report and human-in-the-loop explainability layer. Thanks for watching."

---

**Total: ~2:55** — 5-second buffer for natural pauses.

**Note:** Scenes 3–5 are all live dashboard recordings and can be one continuous take — just switch tabs in order (Drift → Topics → UMAP → Alerts → Model Evaluation).

---

## Recording Checklist

- [ ] Obsidian / other apps minimized before recording
- [ ] Dashboard running: `streamlit run src/app/dashboard.py` (from lab10 root, venv active)
- [ ] Architecture SVG open in browser or image viewer for Scene 2
- [ ] Dashboard pre-loaded on Drift tab before hitting record
- [ ] Sidebar set to Weekly frequency for clearest drift signal
