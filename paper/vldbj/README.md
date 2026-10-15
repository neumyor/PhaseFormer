# PhaseFormer — VLDB Journal extended version

Extension of the ICLR 2026 paper for the VLDBJ special issue "Spatio-Temporal Data
Management and Analytics: Machine Learning-based Solutions and Beyond"
(initial submission due 2026-10-15).

- `main.tex` — manuscript, Springer `svjour3` with the `twocolumn` option, as required by
  the VLDBJ submission rules (two-column template).
- `\blue{...}` (defined in `main.tex`) marks every passage that is new or changed relative
  to the ICLR version. `\showchangesfalse` produces a clean copy.
- `body/` — sections. `theory.tex`, `extension.tex`, `extension_experiments.tex` and
  `appendix_theory.tex` are new; the others start from the ICLR source.
- `iclr_original/` — unmodified ICLR body files, kept for diffing.
- `cover_letter/` — English cover letter (`cover_letter.tex`) and Chinese translation
  (`cover_letter_zh.md`, internal only).
- Class and bibliography styles: `svjour3.cls`, `svglov3.clo`, `spbasic.bst` (Springer
  svjour3 v3.3 package).

Build: `latexmk -pdf main.tex`.
