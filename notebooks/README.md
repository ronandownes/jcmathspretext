# Restricted-Dogs Notebooks

Colab notebooks for the MSc Capstone restricted-breed classifier. Run in order
on Google Colab with the project root mounted at
`/content/drive/MyDrive/MSc_Capstone`.

| # | Notebook | What it does |
|---|---|---|
| 1 | `1_DataPrep_Colab.ipynb` | Downloads Stanford Dogs, identifies the 12 restricted breeds, samples 24 unrestricted breeds, then builds `data/{train,val,test}/<breed>/*.jpg` (36 breed folders per split). Writes `data/breed_to_restricted.json`. |
| 2 | `2_ModelZoo_Colab.ipynb` | Trains 6 backbones with frozen ImageNet weights as a 36-way softmax. Reports both top-1 breed accuracy and binary (restricted vs unrestricted) metrics. |
| 3 | `3_FineTuning_Colab.ipynb` | Hyperband search on InceptionResNetV2, binary threshold tuning, and ensemble of top frozen-base models. |
| 4 | `4_GradCAM_Colab.ipynb` | Grad-CAM heatmaps (one per breed) and most-confident binary misclassifications. |

## Key design choice — multi-class breeds, collapsed to binary

Instead of training a binary `restricted` vs `unrestricted` head directly, the
notebooks train a 36-way softmax over breeds. The binary "restricted"
probability is the **sum of softmax outputs over the 12 restricted breed
indices**.

Reasons:

1. **Stronger gradient signal** — 36-way softmax provides far richer supervision
   than a single bit.
2. **Avoids the smushed-class problem** — "unrestricted" otherwise lumps 24
   visually-distinct breeds into one fuzzy class the model has to learn an
   artificial boundary for.
3. **Better failure analysis** — Notebook 4 can show *which* breed the model
   confused for *which*, not just "wrong".
4. **Same binary deliverable** — the project still produces a binary
   restricted/unrestricted decision (and threshold-tunes it in Notebook 3).

## Other accuracy fixes vs the previous version

- Uses **all** unrestricted images (the old `DIV_UNRESTRICTED = 4` threw away
  75% of the data).
- Per-class **class weights** so per-breed imbalance does not bias training.
- **Per-breed stratified split** (no breed appears in only one split).
- Stronger augmentation (`RandomBrightness`, `RandomContrast`).
- Grad-CAM tap rewritten to backprop from the **sum of restricted softmax
  outputs** (the "is-this-restricted" signal) rather than a single logit, and
  to feed the saved model raw 0–255 images (the previous code double-divided
  by 255 even though the saved model has `preprocess_input` baked in).

## Regenerating

The four `.ipynb` files are produced from `_build_notebooks.py`. To regenerate:

```bash
python3 _build_notebooks.py
```
