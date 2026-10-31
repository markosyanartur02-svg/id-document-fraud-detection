# ID Document Fraud Detection

A deep learning model that flags fraudulent identity documents (driver's licenses and national ID cards), built for the **FreUID Challenge 2026 (IJCAI-ECAI)**.

The model is validated on a **document type it never saw during training**. A fraud detector in production keeps meeting new card designs, so generalising to an unseen template matters more than scoring well on a random split.

![Validation metrics per epoch](assets/validation_metrics.png)

## Results

Validation on the held-out `GUINEA/DL` documents (13,389 images), best epoch:

| ROC-AUC | PR-AUC | Recall | Precision @ 0.5 | Accuracy @ 0.5 |
|---|---|---|---|---|
| **0.999** | **0.998** | **0.999** | 0.692 | 0.821 |

- **Near-perfect ranking on an unseen document type.** Almost every fraudulent card scores higher than almost every genuine one.
- **The default 0.5 threshold is the bottleneck.** On an unfamiliar design the scores shift upward, so about 30% of genuine documents get flagged. Recalibrating the threshold is the next step (see below).

### Shortcut check

An AUC this high on an unseen document type raises a fair question: is the model picking up an artefact, such as fraudulent images being saved at a different size or JPEG quality, instead of the forgery itself? [`shortcut_check.py`](shortcut_check.py) trains a gradient-boosted model on file metadata only (image size, file size, bytes per pixel, JPEG quantisation tables) and tests it on each held-out document type:

| Held-out type | Metadata-only ROC-AUC |
|---|---|
| BENIN/DL | 0.500 |
| EGYPT/DL | 0.625 |
| GUINEA/DL | 0.462 |
| MAURITIUS/ID | 0.357 |
| MOZAMBIQUE/DL | 0.484 |

These scores are close to chance (0.5), so file metadata carries no usable fraud signal. The CNN's 0.999 on GUINEA/DL has to come from the image content.

## Data

- 69,352 synthetic ID document images across 5 document types: Benin, Egypt, Guinea and Mozambique driver's licenses, and Mauritius national ID cards
- Binary label: 1 = fraudulent, 0 = genuine (~42% fraudulent overall)
- The images are not included in this repository; download them from the competition page

## Approach

1. **Leave-one-document-type-out validation.** Every 1- and 2-type hold-out is scored by how close it gets to a 20% validation size and to the overall fraud rate. `GUINEA/DL` wins, giving 55,963 training and 13,389 validation images with no document type shared between them.
2. **Model.** EfficientNet-B0 pretrained on ImageNet, with a single-logit head, fine-tuned end to end.
3. **Training.** `BCEWithLogitsLoss`, Adam (lr 1e-4), batch size 64, mixed precision, 5 epochs on a Kaggle Tesla T4. Images are resized to 224×224 with light colour jitter.
4. **Evaluation.** Accuracy, precision, recall, F1, ROC-AUC and PR-AUC. Recall is the priority metric, since missed fraud is the costly error.

## Next steps

- [ ] Choose the decision threshold on the training types, or calibrate scores, instead of using a fixed 0.5
- [x] Sanity-check for shortcuts: test whether image size or file size alone predicts the label (it doesn't, see above)
- [ ] Full 5-fold leave-one-type-out cross-validation, reporting mean ± std
- [ ] Keep the best checkpoint, add early stopping and a learning-rate schedule
- [ ] Train at higher resolution to keep the fine detail where tampering shows up

## Repository

| File | Contents |
|---|---|
| `id_document_fraud_detection.ipynb` | Full pipeline: data, validation split, model, training, results |
| `shortcut_check.py` | Tests whether file metadata alone predicts fraud |
| `assets/validation_metrics.png` | Per-epoch validation metrics |

## Run it

The notebook runs on Kaggle with a GPU. Attach the FreUID 2026 dataset and update `base` in the first cell if your input path differs.

The shortcut check runs locally in a few seconds, with the labels at `data/train_labels.csv` and the images in `train/`:

```bash
pip install -r requirements.txt
python shortcut_check.py
```

**Stack:** Python, PyTorch, torchvision, scikit-learn, pandas
