# Additional Phase 1 review material

The semester collection contains a 20-slide Phase 1 review deck about comment-aware self-training for Tamil multimodal hate-speech classification. This anonymized summary preserves the relevant method and reported comparison without publishing slides containing team names or student IDs.

## Study setup described in the deck

- 40 training videos and 10 testing videos, paired with transcriptions.
- Four labels: offensive, sexist, racist, and caste-based hate speech.
- Text features from IndicBERT combined with video embeddings.
- Retrieval and an LLM used to propose 10 synthetic examples per original training item (400 proposed examples).
- Pseudo-labeling followed by a student–teacher training workflow.

## Historical comparison

| Model | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Teacher | 43.91 | 48.46 | 46.19 |
| Teacher with synthetic data | 52.41 | 58.23 | 57.68 |
| Student | 66.15 | 70.57 | 68.53 |

Values are transcribed from the review deck. Metric averaging, label mapping, evaluation code, and raw predictions are not established by the slides. These are historical reported values, not independently reproduced benchmarks. The private video dataset is not redistributed.

Audio features appear as proposed future work in the deck. The notebooks remain separate experimental versions; the deck does not establish which saved notebook produced the comparison.

## Source selection

A nearby document named Project Overview discusses general image classification, UAV imagery, and VQA. Its connection to this project is unconfirmed, so its contents are excluded from this showcase.
