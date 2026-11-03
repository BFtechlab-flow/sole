# Technical report

The final report must be written in the organizers' Overleaf template
(<https://www.overleaf.com/read/vjztpvrnbrdh>, ACM `sigconf`, 4 pages
excluding acknowledgment and references; do not alter the black template
text, replace the gray guide text, set `\guideStylefalse` at the end).

Material in this folder:

* `report_draft.tex` — the text for the title, abstract, introduction,
  methodology and evaluation sections of the template, with the measured
  numbers of the final run (2 nov 2026) filled in;
* `references_extra.bib` — bibliography entries to append to the template's
  `main.bib`;
* `figures/pipeline.pdf` — the pipeline figure required by the template
  (regenerate with `python tools/make_pipeline_figure.py`);
* `figures/inputs_and_annotations.png`, `figures/oof_rubric_panels_default.png`,
  `figures/training_curves.png`, `figures/test_predictions.png` — written by
  the notebook in the final run (`artifacts/figures/`).

Rules for the numbers: every value comes from a saved artifact of the final
run (`data_audit.json`, `oof_metrics.json`, `oof_metrics_tuned.json`,
`postprocess_search.json`, `ablations/compare_*.json`,
`submission.verification.json`); the main score is the official micro PQ with
its bootstrap interval; the nested calibration estimate is reported next to
the tuned score; negative ablation results are reported too.
