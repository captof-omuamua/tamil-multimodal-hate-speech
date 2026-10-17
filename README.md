# Tamil Multimodal Hate-Speech Detection

B.Tech final-year Phase 1 research exploring text and video features for Tamil hate-speech classification. Labels in the notebooks are Racist, Caste, Offensive, and Sexist.

## Status

Curated research draft; not a verified runnable release. No independently reproduced benchmark score is claimed. Notebook outputs and execution counts have been cleared in these copies. Original files remain untouched.

## Work explored

- IndicBERT text embeddings and ViT/ResNet video features.
- Feature concatenation and ResNet-18 classification experiments.
- Teacher/student pseudo-labeling.
- Synthetic-data generation and language-model fine-tuning experiments.

## Files

`notebooks/` retains distinct notebook versions for comparison. `scripts/` contains preparation and feature extraction code. `docs/source-map.csv` maps copies to original relative paths and source hashes. Names do not establish which experiment is final.

## Reproduction

There is no validated end-to-end run command yet. Data and trained weights are absent. Scripts contain machine-specific paths and notebooks assume spreadsheets or Colab uploads. Recover data provenance, dependency versions, and the intended model before running. See `docs/REVIEW.md` for release blockers.

Dataset archives, extracted frames, generated embeddings, and the installed Python environment are intentionally excluded. Dataset access terms, team contributions, borrowed template attribution, and project licensing still need documentation. No license has been selected.

## Recovered review material

The [additional Phase 1 review summary](docs/semester-material/README.md) records the study setup and historical model-comparison table recovered from the semester files. Reported figures are distinguished from reproduced evaluations.
