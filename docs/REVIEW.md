# Source review

## Findings

1. In `MultimodalHateSpeechResNet18.ipynb`, `MultimodalResNet.forward()` reshapes embeddings to `(batch, 3, 32, 16)` and applies softmax without calling `self.resnet`. It bypasses the learned network and returns a four-dimensional tensor instead of four-class logits. Saved training results do not validate this implementation.
2. The same notebook passes softmax outputs to cross-entropy and applies another softmax for consistency loss. The corrected implementation needs an explicit logits/probabilities contract.
3. Plotting cells reference `train_loss_teacher` and `train_loss_student` outside the function where those names are defined; returned history variables have different names.
4. Scripts contain absolute Linux paths; notebooks also depend on local spreadsheets or Colab uploads.
5. Original saved results distinguish a 12-sample test set from a 360-sample synthetic test set. Report these separately if reproduced; synthetic performance does not establish equivalent generalization to independent data.
6. `LLM.py` uses a placeholder model identifier and creates feature vectors by adding random noise. Treat it as a prototype rather than verified LLM-generated feature augmentation.
7. Multiple files were byte-identical. One copy per source hash was retained. `phase3.ipynb` is identical to `notFinal_NEW_3_100.ipynb`.

## Before a runnable release

- Compare distinct classifier versions and establish the intended architecture.
- Recover feature dimensions, label alignment, data sources, splits, and team contributions.
- Check synthetic sample provenance and evaluation leakage.
- Replace hardcoded paths with configuration; recover and test dependency versions.
- Correct the model, verify output shapes and gradients, then reproduce training/evaluation with seeds and documented splits.
- Scan code and notebook source for credentials before publishing.

## Separate portfolio candidates

Root-level model.py, train.py, test.py, evaluate.py, and preprocess.py implement Malayalam news summarization. b17.ipynb explores DQN with graph neural networks. The Turkish MLSum notebook is separate. These originals were excluded from this draft and retained at the source.

## Scope

This pass inspected source and notebook cells and compared source hashes. Training was not executed; datasets, archives, reports, licensing, and secrets have not been comprehensively audited. Notebook outputs and widget metadata were removed only from draft copies. No algorithm changes were made.

Credential preparation: a hardcoded Hugging Face token was redacted from the draft copy of scripts/synthetic_2.py. Its source hash in source-map.csv describes the original, not the redacted copy. The original was not modified. Revoke/rotate the original token; configure authentication outside source code. A limited common-token-pattern scan found no remaining matches after redaction.
