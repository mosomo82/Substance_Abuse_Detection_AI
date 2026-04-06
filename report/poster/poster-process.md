# Poster Production Notes

## NSF NRT Research-A-Thon 2026 — Conference Poster

---

## Goal

Produce a print-ready conference poster for the UMKC 2026 Spring Research-A-Thon (April 10).
Final spec: **48" × 36" landscape**, single page, PDF output.

---

## Toolchain

**Rejected approaches:**

- **Single giant SVG** — abandoned early; too hard to iterate on at poster scale
- **LaTeX (xelatex + beamerposter/a0poster)** — TinyTeX 2025 couldn't install packages from the 2026 CTAN repo; `tcolorbox`, `tikz`, `enumitem`, `setspace`, and others all missing. Tried historic 2025 repo but `tlmgr` self-update stalled.
- **cairosvg (Python)** — installed but failed at runtime; Cairo native library (`libcairo-2.dll`) not present on the system.

**What worked:**

- **HTML + CSS → Edge headless PDF**
  - Edge installed at `C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe`
  - `@page { size: 48in 36in; }` sets paper dimensions
  - Rendered with: `msedge --headless --disable-gpu --print-to-pdf=poster.pdf --print-to-pdf-no-header file:///...poster.html`
  - SVG embedded via `<img src="...">` — no conversion needed
  - `svglib` + `reportlab` were also confirmed working for SVG→PDF conversion, but not needed in the end

---

## Architecture Diagram

Used the existing **Track B** pipeline SVG at:
`lab10/data/figures/substance_abuse_ai_pipeline_track_b_architecture.svg`

This has a light/cream background (`#f7f6f1`) which created natural contrast against the poster's off-white body. Embedded as a plain `<img>` tag in a wrapper `div` with a matching background and border.

---

## Layout

Three-column CSS Grid:

- **Left (14in):** Problem statement, datasets table, preprocessing steps, ethics grid
- **Center (18in):** Architecture SVG + four-method ensemble table
- **Right (14in):** Key finding callout, classification results table, clustering stats, RAG example output, future work

Header and footer are full-width bands outside the grid.

---

## Design Decisions

**Color scheme:** Light body (`#f5f4ef`) with dark navy header/footer (`#0e1117`), crimson accent (`#9b2335`). Matched the dark theme of the existing video slides while keeping the body readable for print.

**"Track B" removed** — turned out not to be a real designation; scrubbed from all visible text (filename kept as-is).

**Orientation:** Started portrait (36"×48"), switched to landscape (48"×36") at user request. Required recalculating all column widths and vertical spacing.

---

## Two-Page Problem (and fix)

Edge headless paginates by content flow, not by the CSS `height` property — `overflow: hidden` on `body` doesn't prevent a second page. Fix was two-pronged:

1. Make `grid-template-rows` explicit and summing exactly to 36in: `3.3in 31.65in 1.05in`
2. Set `height` and `overflow: hidden` on `.body` and each column `div`, clamping content to the page boundary

A stray extra `</div>` (introduced during an edit) was also closing the footer early, pushing the GitHub link to page 2. Fixed by removing it.

Edge headless also occasionally hung when called from bash. Workaround: use PowerShell with `-NoProfile -NonInteractive` and `Start-Process ... -Wait`.

---

## Font Size Iteration

Several passes to fill the available space:

| Pass | Body text | Section titles | Notes |
| ---- | --------- | -------------- | ----- |
| Initial | 18pt | 33pt | Too small; ~50% of page empty |
| +1 | 19pt | 30pt | Better but still short |
| +2 | 25pt | 40pt | Good for center; left/right still sparse |
| +3 | 30pt | 48pt | Left/right filled well |
| Header/footer | — | 96pt title | Final pass: header 72→96pt, footer 17→26pt |

---

## Files

| File | Description |
| ---- | ----------- |
| `poster.html` | Source — edit this to iterate |
| `poster.pdf` | Print-ready output (48"×36" landscape) |
| `poster-process.md` | This file |
