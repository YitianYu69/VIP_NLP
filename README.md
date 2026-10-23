# VIP_NLP

Jargon detection and translation for multidisciplinary teams.

**Option A pipeline:** detect discipline-specific jargon spans with BERT, then translate each span into plain language via a glossary.

```
sentence → BERT span detector → glossary lookup → [{jargon, translation, discipline}]
```

## Quick start (team runner)

1. Use the `vip_nlp` conda/kernel env (see project setup).
2. Make sure the saved detector exists at `Notebook/models/jargon_detector/`  
   (created by `option_a_pipeline.ipynb` Step 1).
3. From `Notebook/`:

```bash
python run_option_a.py "The building facade uses cantilevered structures."
```

Or many sentences (one per line):

```bash
python run_option_a.py --file sentences.txt
```

## Main files

| Path | Role |
|---|---|
| `Notebook/option_a_pipeline.ipynb` | Full Option A build: labels, train, eval, inference |
| `Notebook/run_option_a.py` | Simple team inference entry point |
| `Notebook/models/jargon_detector/` | Saved BERT token classifier (local; ~420MB) |
| `Data/final_jargon_dataset.csv` | Source dataset |
| `Data/span_labeled_jargon.csv` | BIO span labels |
| `Data/jargon_glossary.csv` | Jargon → plain-language definitions |

## Notes

- Detector metrics (held-out test): span F1 ≈ **0.89**
- Unknown detected terms return `"translation": null` and `"known": false`
- The BERT detector weights are ignored by Git and are absent from a fresh clone; obtain the saved weights or retrain/save from the notebook before using the runner
- Older sentence-classification / full-sentence rewrite notebooks remain for baselines only
