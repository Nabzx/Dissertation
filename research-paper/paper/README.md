# Paper

First full draft targeting **ALA @ AAMAS** (workshop, non-archival, double-blind).
Format: AAMAS/ACM proceedings; **8 pages excluding references**. Target cycle: ALA 2027
(AAMAS 2027, Hanoi, May 2027; workshop CfP expected with a ~Feb 2027 deadline — verify when
it drops). ALA 2026's deadline (26 Feb 2026) has passed.

## Files
- `main.tex` — the paper (ACM `sigconf`, `anonymous,review` for double-blind submission).
- `references.bib` — bibliography.
- `figures/` — `learning_curves.png` (Fig 2), `alpha_sweep.png` (Fig 3), plus qualitative
  trajectory/heatmap panels. Regenerate the first two with `experiments/plot_curves.py`
  and `experiments/plot_alpha.py`.

## Build (free, on Overleaf)
1. Create a new Overleaf project, upload `main.tex`, `references.bib`, and `figures/`.
2. Set the main document to `main.tex`, compiler **pdfLaTeX**.
3. Compile. Overleaf ships the `acmart` class and `ACM-Reference-Format` style.
4. For camera-ready: remove `anonymous,review` from `\documentclass`, de-anonymise the
   author block, and paste the ALA copyright code where the AAMAS block would go.

## Status / TODO
- [x] End-to-end first draft: abstract, intro, related work, env, method, setup, results,
      discussion, conclusion; Table 1 + Figs 2--4 wired in with real numbers.
- [ ] Fig 1: environment schematic — reuse the dissertation's octagonal-arena render (not
      yet in `results/`); add to `figures/` and reference in Sec. Environment.
- [ ] Supervisor pass (Dr Afzal / Prof Anjum) on framing and claims.
- [ ] Tighten to the 8-page limit once compiled; trim related work if over.
- [ ] Optional: centralised-critic baseline or agent-count sweep if a reviewer wants it.
