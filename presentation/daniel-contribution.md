# Individual Contribution — Daniel Evans
## Lab 10 / NSF NRT Research-A-Thon 2026, Challenge 1

**Role:** Video presentation and demo

---

## Overview

My contribution to the team submission was the 3-minute demo video — from script development through final edit. This involved understanding and accurately representing the full technical system built by the rest of the team, which required substantial iteration to get right.

---

## Script Development

The script was developed collaboratively with an LLM assistant over multiple revision passes. The process was not simply "generate and accept" — it required active review and correction throughout:

- **Visual verification:** When the generated narration didn't match what was actually on screen, I supplied screenshots of the live dashboard for each relevant tab. This caught several meaningful errors — incorrect tab names, descriptions of metrics that weren't visible in the UI, and narration that misrepresented what the charts actually showed (e.g., Scene 5 originally described accuracy/F1 numbers that don't appear in the Model Evaluation tab; Scene 4 originally described the UMAP chart in generic terms that didn't reflect the actual left-to-right risk gradient visible in the data).
- **Manual editing:** After each generated draft, I read through the narration aloud mentally and made edits where the phrasing felt unnatural to say — not incorrect, just awkward in spoken delivery.
- **Accuracy over completeness:** Where I couldn't verify a claim (e.g., my own code contributions to the pipeline), I revised the script to reflect only what I could stand behind. The team summary was corrected to accurately describe my role as presentation rather than development.

---

## Slide Assets

Three supplementary SVG slides were generated to accompany scenes that had no suitable visual from the dashboard or repo:

| Slide | Scene | Content |
|-------|-------|---------|
| `title-card-slide.svg` | Scene 1 | Project title, event, course |
| `cdc-correlation-slide.svg` | Scene 6 | CDC correlation finding (r=0.471, p=0.006, lag −3 weeks) |
| `team-summary-slide.svg` | Scene 7 | Team member cards with role descriptions |

These were generated via LLM and refined iteratively — adjustments included layout, removing unnecessary footer content, and correcting the "Track B" designation that wasn't in the Canvas assignment.

---

## Recording

**Software:** OBS Studio (screen capture + microphone)

Recording was done scene by scene. Most scenes required one primary pass, but several received an additional pass after re-reading the script:

- Some scenes were rerecorded due to script misalignment discovered only during delivery
- At least one scene was rerecorded because the wrong content was on screen at the time of recording

---

## Editing

**Software:** Shotcut

The edit was straightforward — scene assembly, trimming dead air, and ensuring clean cuts between the static SVG slides and live dashboard segments. Shotcut handled this with minimal friction.

---

## Files Produced

| File | Description |
|------|-------------|
| `video-script.md` | Final scene-by-scene narration script |
| `title-card-slide.svg` | SVG title card for Scene 1 |
| `cdc-correlation-slide.svg` | SVG findings slide for Scene 6 |
| `team-summary-slide.svg` | SVG team summary slide for Scene 7 |
| [video](https://drive.google.com/file/d/14KaZGUeLJig0ayF_UU8aXy2XPABRFBxF) | Final edited 3-minute video, submitted to Canvas and Research-A-Thon |

The above files (excluding video) were committed in `ede0d7b` — `feat: add presentation slides and video script for Lab 10 demo`.
