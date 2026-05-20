"""Generate the four Colab notebooks for the restricted-dogs project.

Run this once to (re)write the .ipynb files in this directory.

Design change vs earlier versions:
- One folder per breed (36 classes) instead of binary restricted/unrestricted
- Multi-class softmax during training; collapse to binary at inference by
  summing softmax probabilities over the restricted breed indices
- Keeps ALL unrestricted images (no 1/4 sub-sampling) + class weights
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

OUT_DIR = Path(__file__).parent


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in textwrap.dedent(text).strip("\n").splitlines()],
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [line + "\n" for line in textwrap.dedent(text).strip("\n").splitlines()],
    }


def write_notebook(name: str, cells: list[dict]) -> None:
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
            "colab": {"provenance": []},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = OUT_DIR / name
    path.write_text(json.dumps(nb, indent=1))
    print(f"wrote {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Common preamble (Drive mount + GPU + paths). Shared across notebooks.
# ─────────────────────────────────────────────────────────────────────────────
PREAMBLE = [
    code(
        """
        # ── Mount Google Drive ─────────────────────────────────────
        from google.colab import drive
        drive.mount('/content/drive')
        print('Drive mounted.')
        """
    ),
    code(
        """
        # ── Verify GPU ────────────────────────────────────────────
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f'GPU detected: {gpus[0].name}')
            tf.config.experimental.set_memory_growth(gpus[0], True)
        else:
            print('No GPU — go to Runtime -> Change runtime type -> T4 GPU')
        """
    ),
    code(
        """
        # ── Project root on Drive ─────────────────────────────────
        from pathlib import Path

        ROOT = Path('/content/drive/MyDrive/MSc_Capstone')
        for d in ['data', 'pretrained/checkpoints', 'pretrained/logs',
                  'pretrained/curves', 'pretrained/inference', 'report']:
            (ROOT / d).mkdir(parents=True, exist_ok=True)

        DATA   = ROOT / 'data'
        CKPT   = ROOT / 'pretrained' / 'checkpoints'
        LOGS   = ROOT / 'pretrained' / 'logs'
        CURVES = ROOT / 'pretrained' / 'curves'
        INFER  = ROOT / 'pretrained' / 'inference'
        REPORT = ROOT / 'report'

        print(f'Project root: {ROOT}')
        print(f'Data folder:  {DATA}')
        """
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
# Notebook 1 — Dataset Preparation (one folder per breed)
# ─────────────────────────────────────────────────────────────────────────────
nb1_cells: list[dict] = [
    md(
        """
        # Notebook 1 — Dataset Preparation (one folder per breed)

        Builds `data/{train,val,test}/<breed>/*.jpg` with **36 breed folders** total
        (12 restricted + 24 unrestricted). A JSON manifest `breed_to_restricted.json`
        records which breeds are restricted, so later notebooks can collapse
        the multi-class softmax to a binary "restricted vs unrestricted" decision.

        Idempotent: each step skips itself if its output is already correct.
        """
    ),
    *PREAMBLE,
    md("## 1.1 — Download Stanford Dogs (skip if already done)"),
    code(
        """
        import urllib.request

        TAR_PATH   = DATA / 'images.tar'
        IMAGES_DIR = DATA / 'images'
        URL = 'http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar'

        if not TAR_PATH.exists():
            print('Downloading Stanford Dogs (~800MB)...')
            urllib.request.urlretrieve(URL, TAR_PATH)
            print('Download complete.')
        else:
            print('images.tar exists — skipping download.')
        """
    ),
    md("## 1.2 — Extract"),
    code(
        """
        import tarfile

        if not IMAGES_DIR.exists():
            capital_I = DATA / 'Images'
            if capital_I.exists():
                capital_I.rename(IMAGES_DIR)
            else:
                print('Extracting...')
                with tarfile.open(TAR_PATH, 'r') as tar:
                    tar.extractall(DATA, filter='data')
                capital_I = DATA / 'Images'
                if capital_I.exists():
                    capital_I.rename(IMAGES_DIR)
                print('Extraction complete.')
        else:
            print('images/ exists — skipping extraction.')

        print(f'Breeds found: {len(list(IMAGES_DIR.iterdir()))}')
        """
    ),
    md("## 1.3 — Normalise breed folder names"),
    code(
        """
        import re, shutil

        first_dir = next(IMAGES_DIR.iterdir())
        already_normalised = not bool(re.match(r'^n[0-9]+-', first_dir.name))

        if already_normalised:
            print('Already normalised — skipping.')
        else:
            def normalise(breed_dir):
                slug = re.sub(r'^n[0-9]+-', '', breed_dir.name).lower().replace('-', '_')
                new_dir = breed_dir.parent / slug
                if breed_dir != new_dir:
                    if new_dir.exists(): shutil.rmtree(new_dir)
                    breed_dir.rename(new_dir)
                    breed_dir = new_dir
                for i, img in enumerate(sorted(breed_dir.glob('*'))):
                    if img.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
                        img.rename(breed_dir / f'{slug}_{i+1:03d}{img.suffix.lower()}')
            for d in IMAGES_DIR.iterdir():
                if d.is_dir(): normalise(d)
            print(f'Normalised {len(list(IMAGES_DIR.iterdir()))} breeds.')

        ALL_BREEDS = sorted(d.name for d in IMAGES_DIR.iterdir() if d.is_dir())
        print(f'Total breeds: {len(ALL_BREEDS)}')
        """
    ),
    md(
        """
        ## 1.4 — Identify restricted breeds (keyword match)

        Stanford folder names don't always match the legal breed names exactly
        (`german_shepherd` vs `german_shepherd_dog`), so we match by keyword.
        """
    ),
    code(
        """
        RESTRICTED_KEYWORDS = [
            (['bull_mastiff', 'bullmastiff'],           'Bull Mastiff'),
            (['doberman'],                              'Doberman Pinscher'),
            (['german_shepherd'],                       'German Shepherd'),
            (['rhodesian_ridgeback'],                   'Rhodesian Ridgeback'),
            (['rottweiler'],                            'Rottweiler'),
            (['staffordshire_bull', 'staffordshire_bullterrier'], 'Staffordshire Bull Terrier'),
            (['american_pit', 'pit_bull'],              'American Pit Bull Terrier'),
            (['english_bull_terrier', 'bull_terrier'],  'English Bull Terrier'),
            (['japanese_akita', 'akita'],               'Japanese Akita'),
            (['japanese_tosa', 'tosa'],                 'Japanese Tosa'),
            (['bandog'],                                'Bandog'),
            (['xl_bully', 'american_bully'],            'XL Bully'),
        ]

        RESTRICTED = set()
        print('Restricted breeds present in Stanford Dogs:')
        for keywords, display in RESTRICTED_KEYWORDS:
            matches = [b for b in ALL_BREEDS if any(k in b for k in keywords)]
            if matches:
                for m in matches:
                    RESTRICTED.add(m)
                    n = len(list((IMAGES_DIR / m).glob('*.jpg')))
                    print(f'  {display:30s} -> {m} ({n} images)')
            else:
                print(f'  {display:30s} -> NOT FOUND')

        print(f'\\nTotal restricted breeds matched: {len(RESTRICTED)} of 12')
        """
    ),
    md(
        """
        ## 1.5 — Pick 24 unrestricted breeds (deterministic) and assemble the breed list

        We use all available restricted breeds and randomly sample 24 unrestricted
        breeds (with a fixed seed) so the experiment is reproducible. The full
        breed list — 36 entries — is saved alongside `is_restricted` flags so
        every notebook downstream uses the same mapping.
        """
    ),
    code(
        """
        import json, random

        SEED = 58
        random.seed(SEED)

        pool = sorted(set(ALL_BREEDS) - RESTRICTED)
        UNRESTRICTED = sorted(random.sample(pool, min(24, len(pool))))

        SELECTED_BREEDS = sorted(RESTRICTED) + UNRESTRICTED
        BREED_TO_RESTRICTED = {b: (b in RESTRICTED) for b in SELECTED_BREEDS}

        manifest_path = DATA / 'breed_to_restricted.json'
        with open(manifest_path, 'w') as f:
            json.dump({
                'breeds': SELECTED_BREEDS,
                'restricted': sorted(RESTRICTED),
                'unrestricted': UNRESTRICTED,
                'breed_to_restricted': BREED_TO_RESTRICTED,
                'seed': SEED,
            }, f, indent=2)

        print(f'{len(SELECTED_BREEDS)} breeds selected ({len(RESTRICTED)} restricted + {len(UNRESTRICTED)} unrestricted).')
        print(f'Manifest written: {manifest_path}')
        """
    ),
    md(
        """
        ## 1.6 — Build `data/{train,val,test}/<breed>/` (stratified 60/20/20)

        Each breed becomes its own folder under each split. Stratification is
        per-breed, so every breed appears in train, val, and test in the same
        60/20/20 proportion. Re-running is safe: if the layout is already
        correct it skips straight to verification.
        """
    ),
    code(
        """
        import shutil

        SPLITS = ('train', 'val', 'test')

        def layout_is_valid():
            for split in SPLITS:
                split_dir = DATA / split
                if not split_dir.is_dir(): return False
                existing = {d.name for d in split_dir.iterdir() if d.is_dir()}
                if existing != set(SELECTED_BREEDS):
                    return False
                for breed in SELECTED_BREEDS:
                    if not any((split_dir / breed).glob('*.jpg')):
                        return False
            return True

        if layout_is_valid():
            print('Splits already valid — skipping rebuild.')
        else:
            # Wipe the old layout (binary or partial) before rebuilding.
            for legacy in ['restricted', 'unrestricted']:
                p = DATA / legacy
                if p.exists():
                    print(f'Removing legacy {p}')
                    shutil.rmtree(p)
            for split in SPLITS:
                p = DATA / split
                if p.exists():
                    print(f'Removing old split: {p}')
                    shutil.rmtree(p)
                for breed in SELECTED_BREEDS:
                    (p / breed).mkdir(parents=True)

            def split_counts(n):
                t = int(n * 0.6); v = int(n * 0.2)
                return t, v, n - t - v

            random.seed(SEED)
            summary = {s: 0 for s in SPLITS}
            per_breed = {}

            for breed in SELECTED_BREEDS:
                imgs = sorted((IMAGES_DIR / breed).glob('*.jpg'))
                random.shuffle(imgs)
                t, v, te = split_counts(len(imgs))
                parts = {'train': imgs[:t], 'val': imgs[t:t+v], 'test': imgs[t+v:]}
                per_breed[breed] = {s: len(parts[s]) for s in SPLITS}
                for split, files in parts.items():
                    for f in files:
                        shutil.copy2(f, DATA / split / breed / f.name)
                        summary[split] += 1

            print(f'\\n{"Split":<6} | {"Images":>7}')
            print('-' * 18)
            for s in SPLITS:
                print(f'{s:<6} | {summary[s]:>7}')
            print(f'\\nPer-breed counts (train/val/test):')
            for breed, counts in per_breed.items():
                tag = 'R' if BREED_TO_RESTRICTED[breed] else 'U'
                print(f'  [{tag}] {breed:30s} {counts["train"]:>4}/{counts["val"]:>3}/{counts["test"]:>3}')
        """
    ),
    md("## 1.7 — Final verification"),
    code(
        """
        print('FINAL VERIFICATION\\n')
        ok = True
        totals = {'restricted': 0, 'unrestricted': 0}
        for split in SPLITS:
            r = u = 0
            for breed in SELECTED_BREEDS:
                p = DATA / split / breed
                if not p.is_dir():
                    print(f'MISSING: {p}'); ok = False; continue
                n = len(list(p.glob('*.jpg')))
                if n == 0:
                    print(f'EMPTY:   {p}'); ok = False
                if BREED_TO_RESTRICTED[breed]:
                    r += n
                else:
                    u += n
            totals['restricted'] += r
            totals['unrestricted'] += u
            print(f'{split.upper():5s}  restricted={r:>4}  unrestricted={u:>4}  total={r+u:>4}')

        print(f'\\nOverall: restricted={totals["restricted"]}, unrestricted={totals["unrestricted"]}')
        print('ALL CHECKS PASSED.' if ok else 'PROBLEMS DETECTED — check above.')
        if ok:
            print('\\nNotebook 1 complete. Open Notebook 2 and Run all.')
        """
    ),
]
write_notebook("1_DataPrep_Colab.ipynb", nb1_cells)


# ─────────────────────────────────────────────────────────────────────────────
# Shared helper cell text used by notebooks 2, 3, 4
# ─────────────────────────────────────────────────────────────────────────────
MANIFEST_LOAD = code(
    """
    # ── Load breed manifest written by Notebook 1 ────────────────
    import json, numpy as np

    with open(DATA / 'breed_to_restricted.json') as f:
        MANIFEST = json.load(f)

    SELECTED_BREEDS = MANIFEST['breeds']
    RESTRICTED_SET  = set(MANIFEST['restricted'])
    NUM_CLASSES     = len(SELECTED_BREEDS)
    BREED_TO_IDX    = {b: i for i, b in enumerate(SELECTED_BREEDS)}

    # Boolean mask aligned with class index order used by image_dataset_from_directory
    # (alphabetical). image_dataset_from_directory sorts class names alphabetically,
    # so we mirror that ordering here.
    CLASS_NAMES     = sorted(SELECTED_BREEDS)
    RESTRICTED_MASK = np.array([(b in RESTRICTED_SET) for b in CLASS_NAMES], dtype=bool)
    RESTRICTED_IDX  = np.where(RESTRICTED_MASK)[0]

    print(f'{NUM_CLASSES} classes loaded.')
    print(f'Restricted indices (in alphabetical order): {RESTRICTED_IDX.tolist()}')
    """
)


# ─────────────────────────────────────────────────────────────────────────────
# Notebook 2 — Model Zoo
# ─────────────────────────────────────────────────────────────────────────────
nb2_cells: list[dict] = [
    md(
        """
        # Notebook 2 — Model Zoo (multi-class breed → binary)

        Train six CNN backbones with frozen ImageNet weights on the 36-breed
        classification task. Per-model metrics are reported in **both**
        flavours:

        * **Multi-class** (top-1 breed accuracy) — what the model is optimising
        * **Binary** (restricted vs unrestricted) — what the project actually needs

        The binary probability is computed by summing softmax probabilities
        over the restricted breed indices: `p_restricted = sum(softmax[RESTRICTED_IDX])`.
        """
    ),
    *PREAMBLE,
    md("## 2.1 — Configuration and manifest"),
    code(
        """
        from tensorflow.keras import layers
        import numpy as np, pandas as pd, json, random
        import matplotlib.pyplot as plt, seaborn as sns
        from sklearn.metrics import confusion_matrix, roc_auc_score

        SEED = 58
        tf.random.set_seed(SEED); np.random.seed(SEED); random.seed(SEED)

        BATCH_SIZE    = 16
        EPOCHS        = 22
        PATIENCE      = 3
        LEARNING_RATE = 1e-4

        print('Config set.')
        """
    ),
    MANIFEST_LOAD,
    md("## 2.2 — Dataset loader (multi-class, with class weights)"),
    code(
        """
        from sklearn.utils.class_weight import compute_class_weight

        def create_datasets(input_size=(224, 224)):
            augmentation = tf.keras.Sequential([
                layers.RandomFlip('horizontal'),
                layers.RandomRotation(0.05),
                layers.RandomZoom(0.1),
                layers.RandomBrightness(0.1),
                layers.RandomContrast(0.1),
            ], name='augmentation')

            train_ds = tf.keras.utils.image_dataset_from_directory(
                str(DATA / 'train'), image_size=input_size, batch_size=BATCH_SIZE,
                label_mode='int', class_names=CLASS_NAMES,
                shuffle=True, seed=SEED)
            val_ds = tf.keras.utils.image_dataset_from_directory(
                str(DATA / 'val'), image_size=input_size, batch_size=BATCH_SIZE,
                label_mode='int', class_names=CLASS_NAMES, shuffle=False)
            test_ds = tf.keras.utils.image_dataset_from_directory(
                str(DATA / 'test'), image_size=input_size, batch_size=BATCH_SIZE,
                label_mode='int', class_names=CLASS_NAMES, shuffle=False)

            # Class weights for the multi-class loss (handles breed imbalance).
            y_train = np.concatenate([y.numpy() for _, y in train_ds])
            cw = compute_class_weight('balanced',
                                      classes=np.arange(NUM_CLASSES), y=y_train)
            class_weight = {i: float(w) for i, w in enumerate(cw)}

            train_ds = train_ds.map(lambda x, y: (augmentation(x, training=True), y))
            AUTOTUNE = tf.data.AUTOTUNE
            return (
                train_ds.cache().shuffle(1000).prefetch(AUTOTUNE),
                val_ds.cache().prefetch(AUTOTUNE),
                test_ds.cache().prefetch(AUTOTUNE),
                class_weight,
            )

        print('Dataset loader ready.')
        """
    ),
    md("## 2.3 — Model registry"),
    code(
        """
        from tensorflow.keras.applications import (
            vgg16, resnet50, inception_v3, xception,
            inception_resnet_v2, nasnet
        )

        MODEL_ZOO = {
            'VGG16':             (vgg16.VGG16,                          vgg16.preprocess_input,              (224, 224)),
            'ResNet50':          (resnet50.ResNet50,                     resnet50.preprocess_input,           (224, 224)),
            'InceptionV3':       (inception_v3.InceptionV3,              inception_v3.preprocess_input,       (299, 299)),
            'Xception':          (xception.Xception,                     xception.preprocess_input,           (299, 299)),
            'InceptionResNetV2': (inception_resnet_v2.InceptionResNetV2, inception_resnet_v2.preprocess_input,(299, 299)),
            'NASNetMobile':      (nasnet.NASNetMobile,                   nasnet.preprocess_input,             (224, 224)),
        }

        def build_model(name):
            constructor, preprocess_fn, input_size = MODEL_ZOO[name]
            base = constructor(include_top=False, weights='imagenet',
                               input_shape=(*input_size, 3))
            base.trainable = False
            inputs = tf.keras.Input(shape=(*input_size, 3))
            x = preprocess_fn(inputs)
            x = base(x, training=False)
            x = layers.GlobalAveragePooling2D()(x)
            x = layers.Dropout(0.3)(x)
            outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
            model = tf.keras.Model(inputs, outputs)
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                loss='sparse_categorical_crossentropy',
                metrics=[
                    tf.keras.metrics.SparseCategoricalAccuracy(name='top1'),
                    tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top5'),
                ],
            )
            total  = model.count_params()
            frozen = sum(tf.size(w).numpy() for w in base.non_trainable_weights)
            print(f'  {name}: {total:,} params ({frozen:,} frozen, {total-frozen:,} trainable)')
            return model

        print('Models available:', list(MODEL_ZOO.keys()))
        """
    ),
    md("## 2.4 — Train all models"),
    code(
        """
        class CSVLogger(tf.keras.callbacks.Callback):
            def __init__(self, name):
                super().__init__()
                self.path = LOGS / f'{name}_training_log.csv'
                self.rows = []
            def on_epoch_end(self, epoch, logs=None):
                self.rows.append({'epoch': epoch+1, **(logs or {})})
                pd.DataFrame(self.rows).to_csv(self.path, index=False)

        def get_callbacks(name):
            return [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=PATIENCE,
                    restore_best_weights=True, verbose=1),
                tf.keras.callbacks.ModelCheckpoint(
                    str(CKPT / f'{name}_best.keras'),
                    monitor='val_loss', save_best_only=True, verbose=1),
                CSVLogger(name),
            ]

        TO_RUN = ['VGG16', 'ResNet50', 'InceptionV3', 'Xception',
                  'InceptionResNetV2', 'NASNetMobile']

        all_results = []
        all_histories = {}

        for name in TO_RUN:
            print(f'\\n{"="*60}\\n  Training {name}\\n{"="*60}')

            _, _, input_size = MODEL_ZOO[name]
            train_ds, val_ds, test_ds, class_weight = create_datasets(input_size)
            model = build_model(name)

            history = model.fit(
                train_ds, epochs=EPOCHS, validation_data=val_ds,
                callbacks=get_callbacks(name), class_weight=class_weight,
                verbose=1,
            )
            all_histories[name] = history.history

            # ---- Multi-class test metrics ----
            test_results = model.evaluate(test_ds, verbose=0)
            metrics = dict(zip(model.metrics_names, test_results))

            # ---- Binary test metrics (collapse softmax → restricted prob) ----
            y_true_mc = np.concatenate([y.numpy() for _, y in test_ds])
            y_prob_mc = model.predict(test_ds, verbose=0)
            y_prob_bin = y_prob_mc[:, RESTRICTED_IDX].sum(axis=1)
            y_true_bin = RESTRICTED_MASK[y_true_mc].astype(int)
            y_pred_bin = (y_prob_bin > 0.5).astype(int)

            cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
            TP, TN, FP, FN = cm[1,1], cm[0,0], cm[0,1], cm[1,0]
            metrics['bin_accuracy']  = (TP+TN) / max(TP+TN+FP+FN, 1)
            metrics['bin_precision'] = TP / max(TP+FP, 1)
            metrics['bin_recall']    = TP / max(TP+FN, 1)
            metrics['bin_f1']        = 2*TP / max(2*TP+FP+FN, 1)
            metrics['bin_auc']       = roc_auc_score(y_true_bin, y_prob_bin)
            metrics['model']         = name
            metrics['epochs_trained']= len(history.history['loss'])
            all_results.append(metrics)
            print(f'  {name}: top1={metrics["top1"]:.3f}  '
                  f'bin_acc={metrics["bin_accuracy"]:.3f}  '
                  f'bin_f1={metrics["bin_f1"]:.3f}  '
                  f'bin_auc={metrics["bin_auc"]:.3f}')

            del model
            tf.keras.backend.clear_session()

        pd.DataFrame(all_results).to_csv(LOGS / 'MASTER_results.csv', index=False)
        with open(LOGS / 'MASTER_results.json', 'w') as f:
            json.dump(all_results, f, indent=2)
        np.save(LOGS / 'training_histories.npy', all_histories, allow_pickle=True)

        print('\\nAll models trained.')
        """
    ),
    md("## 2.5 — Learning curves"),
    code(
        """
        for name, hist in all_histories.items():
            fig, ax = plt.subplots(figsize=(10, 6))
            epochs = range(1, len(hist['top1']) + 1)
            ax.plot(epochs, hist['top1'],     '#1f77b4', ls='-',  lw=2, label='Train top-1')
            ax.plot(epochs, hist['val_top1'], '#1f77b4', ls='--', lw=2, label='Val top-1')
            ax.plot(epochs, hist['loss'],     '#ff7f0e', ls='-',  lw=2, label='Train loss')
            ax.plot(epochs, hist['val_loss'], '#ff7f0e', ls='--', lw=2, label='Val loss')
            ax.set_title(f'{name} — Learning Curves', fontsize=14, fontweight='bold')
            ax.set_xlabel('Epoch'); ax.set_ylabel('Value')
            ax.legend(); ax.grid(True, alpha=0.3); ax.set_ylim(bottom=0)
            plt.tight_layout()
            plt.savefig(str(CURVES / f'{name}_learning_curves.png'), dpi=300, facecolor='white')
            plt.show(); plt.close()
        """
    ),
    md("## 2.6 — Confusion matrices (binary view) and metrics comparison"),
    code(
        """
        metrics_list = []

        for name in TO_RUN:
            _, _, input_size = MODEL_ZOO[name]
            test_ds = tf.keras.utils.image_dataset_from_directory(
                str(DATA / 'test'), image_size=input_size, batch_size=BATCH_SIZE,
                label_mode='int', class_names=CLASS_NAMES, shuffle=False)
            model_path = CKPT / f'{name}_best.keras'
            if not model_path.exists(): continue
            model = tf.keras.models.load_model(str(model_path))

            y_true_mc = np.concatenate([y.numpy() for _, y in test_ds])
            y_prob_mc = model.predict(test_ds, verbose=0)
            y_prob_bin = y_prob_mc[:, RESTRICTED_IDX].sum(axis=1)
            y_true_bin = RESTRICTED_MASK[y_true_mc].astype(int)
            y_pred_bin = (y_prob_bin > 0.5).astype(int)

            top1 = float(np.mean(y_prob_mc.argmax(axis=1) == y_true_mc))
            cm   = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
            TP, TN, FP, FN = cm[1,1], cm[0,0], cm[0,1], cm[1,0]
            acc  = (TP+TN) / max(TP+TN+FP+FN, 1)
            prec = TP / max(TP+FP, 1)
            rec  = TP / max(TP+FN, 1)
            f1   = 2*TP / max(2*TP+FP+FN, 1)
            auc_v = roc_auc_score(y_true_bin, y_prob_bin)
            metrics_list.append({'model': name, 'top1_breed': top1,
                                 'accuracy': acc, 'precision': prec,
                                 'recall': rec, 'f1': f1, 'auc': auc_v})

            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=['Unrestricted','Restricted'],
                        yticklabels=['Unrestricted','Restricted'])
            ax.set_title(f'{name} (binary view)', fontsize=14, fontweight='bold')
            ax.set_ylabel('True'); ax.set_xlabel('Predicted')
            plt.tight_layout()
            plt.savefig(str(INFER / f'{name}_confusion.png'), dpi=300, facecolor='white')
            plt.show(); plt.close()
            print(f'{name}: top1={top1:.3f}  Acc={acc:.3f}  P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}  AUC={auc_v:.3f}')

            del model
            tf.keras.backend.clear_session()

        metrics_df = pd.DataFrame(metrics_list)
        metrics_df.to_csv(INFER / 'all_models_metrics.csv', index=False)
        print('\\n', metrics_df.round(4).to_string(index=False))
        print('\\nNotebook 2 complete. Open Notebook 3 and Run all.')
        """
    ),
]
write_notebook("2_ModelZoo_Colab.ipynb", nb2_cells)


# ─────────────────────────────────────────────────────────────────────────────
# Notebook 3 — Fine-tuning + threshold + ensemble
# ─────────────────────────────────────────────────────────────────────────────
nb3_cells: list[dict] = [
    md(
        """
        # Notebook 3 — Fine-Tuning, Threshold, and Ensemble (multi-class → binary)

        Hyperband over InceptionResNetV2 (still multi-class softmax over the
        36 breeds), then threshold-tune the *binary* decision (sum of restricted
        softmax probabilities), then ensemble the top frozen-base models.
        """
    ),
    *PREAMBLE,
    md("## 3.1 — Install Keras Tuner and setup"),
    code(
        """
        !pip install -q keras-tuner

        import keras_tuner as kt
        from tensorflow.keras import layers
        from tensorflow.keras.applications import inception_resnet_v2
        import numpy as np, pandas as pd, json, random
        import matplotlib.pyplot as plt, seaborn as sns
        from sklearn.metrics import (
            confusion_matrix, classification_report,
            roc_curve, auc, precision_recall_curve,
            matthews_corrcoef, average_precision_score, roc_auc_score
        )

        SEED = 58
        np.random.seed(SEED); tf.random.set_seed(SEED); random.seed(SEED)

        IMG_SIZE   = (256, 256)
        BATCH_SIZE = 32
        EPOCHS     = 8

        print('Ready for hyperparameter search.')
        """
    ),
    MANIFEST_LOAD,
    md("## 3.2 — Dataset loader"),
    code(
        """
        from sklearn.utils.class_weight import compute_class_weight

        def create_datasets(input_size=(256, 256)):
            aug = tf.keras.Sequential([
                layers.RandomFlip('horizontal'),
                layers.RandomRotation(0.05),
                layers.RandomZoom(0.1),
                layers.RandomBrightness(0.1),
                layers.RandomContrast(0.1),
            ])
            train_ds = tf.keras.utils.image_dataset_from_directory(
                str(DATA / 'train'), image_size=input_size, batch_size=BATCH_SIZE,
                label_mode='int', class_names=CLASS_NAMES, shuffle=True, seed=SEED)
            val_ds = tf.keras.utils.image_dataset_from_directory(
                str(DATA / 'val'), image_size=input_size, batch_size=BATCH_SIZE,
                label_mode='int', class_names=CLASS_NAMES, shuffle=False)
            test_ds = tf.keras.utils.image_dataset_from_directory(
                str(DATA / 'test'), image_size=input_size, batch_size=BATCH_SIZE,
                label_mode='int', class_names=CLASS_NAMES, shuffle=False)

            y_train = np.concatenate([y.numpy() for _, y in train_ds])
            cw = compute_class_weight('balanced',
                                      classes=np.arange(NUM_CLASSES), y=y_train)
            class_weight = {i: float(w) for i, w in enumerate(cw)}

            train_ds = train_ds.map(lambda x, y: (aug(x, training=True), y))
            A = tf.data.AUTOTUNE
            return (
                train_ds.cache().shuffle(1000).prefetch(A),
                val_ds.cache().prefetch(A),
                test_ds.cache().prefetch(A),
                class_weight,
            )

        train_ds, val_ds, test_ds, class_weight = create_datasets()
        print('Datasets loaded.')
        """
    ),
    md("## 3.3 — HyperModel"),
    code(
        """
        def build_model(hp):
            dropout  = hp.Choice('dropout', [0.2, 0.3, 0.4, 0.5])
            lr       = hp.Choice('lr', [3e-5, 1e-4, 3e-4])
            unfreeze = hp.Choice('unfreeze_pct', [0.0, 0.1, 0.2, 0.3])

            base = inception_resnet_v2.InceptionResNetV2(
                include_top=False, weights='imagenet',
                input_shape=(*IMG_SIZE, 3))
            base.trainable = False
            if unfreeze > 0:
                cut = int(len(base.layers) * (1 - unfreeze))
                for layer in base.layers[cut:]:
                    layer.trainable = True

            inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
            x = inception_resnet_v2.preprocess_input(inputs)
            x = base(x, training=True)
            x = layers.GlobalAveragePooling2D()(x)
            x = layers.Dropout(dropout)(x)
            outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

            model = tf.keras.Model(inputs, outputs)
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                loss='sparse_categorical_crossentropy',
                metrics=[
                    tf.keras.metrics.SparseCategoricalAccuracy(name='top1'),
                    tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top5'),
                ],
            )
            return model

        print('HyperModel defined.')
        """
    ),
    md("## 3.4 — Hyperband search"),
    code(
        """
        tuner = kt.Hyperband(
            build_model,
            objective='val_top1',
            max_epochs=EPOCHS,
            factor=4,
            directory=str(ROOT / 'pretrained' / 'hyperband'),
            project_name='restricted_dogs_breeds'
        )

        stop_early = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=2, restore_best_weights=True)

        print('Starting Hyperband search...\\n')
        tuner.search(train_ds, validation_data=val_ds, epochs=EPOCHS,
                     callbacks=[stop_early], class_weight=class_weight, verbose=2)

        best_hp = tuner.get_best_hyperparameters(1)[0]
        print('\\nBest hyperparameters:')
        print(f'  Dropout:       {best_hp.get("dropout")}')
        print(f'  Learning rate: {best_hp.get("lr")}')
        print(f'  Unfreeze %:    {best_hp.get("unfreeze_pct")}')

        best_model = tuner.get_best_models(1)[0]
        best_model.save(str(CKPT / 'FineTuned_best.keras'))
        print(f'Saved: {CKPT / "FineTuned_best.keras"}')
        """
    ),
    md("## 3.5 — Threshold optimisation on the *binary* prediction"),
    code(
        """
        y_true_mc = np.concatenate([y.numpy() for _, y in test_ds])
        y_prob_mc = best_model.predict(test_ds).reshape(-1, NUM_CLASSES)
        y_prob    = y_prob_mc[:, RESTRICTED_IDX].sum(axis=1)
        y_true    = RESTRICTED_MASK[y_true_mc].astype(int)

        thresholds = np.arange(0.1, 0.91, 0.05)
        rows = []
        for t in thresholds:
            y_pred = (y_prob >= t).astype(int)
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            TP, TN, FP, FN = cm[1,1], cm[0,0], cm[0,1], cm[1,0]
            prec = TP / max(TP+FP, 1)
            rec  = TP / max(TP+FN, 1)
            f1   = 2*TP / max(2*TP+FP+FN, 1)
            mcc  = matthews_corrcoef(y_true, y_pred)
            rows.append({'threshold': round(t,2), 'precision': prec,
                         'recall': rec, 'f1': f1, 'mcc': mcc})

        df_t = pd.DataFrame(rows)
        best_row = df_t.loc[df_t['f1'].idxmax()]
        print(f'Optimal threshold: {best_row["threshold"]}  '
              f'F1={best_row["f1"]:.3f}  P={best_row["precision"]:.3f}  '
              f'R={best_row["recall"]:.3f}  MCC={best_row["mcc"]:.3f}')

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df_t['threshold'], df_t['precision'], 'g-', lw=2, label='Precision')
        ax.plot(df_t['threshold'], df_t['recall'],    'b-', lw=2, label='Recall')
        ax.plot(df_t['threshold'], df_t['f1'],        'r-', lw=2.5, label='F1')
        ax.axvline(best_row['threshold'], color='gray', ls='--', alpha=0.7,
                   label=f'Optimal ({best_row["threshold"]})')
        ax.set_xlabel('Threshold on P(restricted)'); ax.set_ylabel('Score')
        ax.set_title('Threshold Optimisation (binary)', fontsize=14, fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(str(INFER / 'threshold_optimisation.png'), dpi=300)
        plt.show()
        df_t.to_csv(str(INFER / 'threshold_sweep.csv'), index=False)
        """
    ),
    md("## 3.6 — ROC, PR, and classification report (binary)"),
    code(
        """
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc_val = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, lw=2, label=f'AUC = {roc_auc_val:.3f}')
        ax.plot([0,1],[0,1], '--', color='gray')
        ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve — Fine-Tuned (binary view)', fontsize=14, fontweight='bold')
        ax.legend(); plt.tight_layout()
        plt.savefig(str(INFER / 'roc_curve.png'), dpi=300); plt.show()

        prec_c, rec_c, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(rec_c, prec_c, lw=2, label=f'PR-AUC = {pr_auc:.3f}')
        ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        ax.legend(); plt.tight_layout()
        plt.savefig(str(INFER / 'pr_curve.png'), dpi=300); plt.show()

        opt_t = best_row['threshold']
        y_opt = (y_prob >= opt_t).astype(int)
        report = classification_report(y_true, y_opt,
                                       target_names=['Unrestricted', 'Restricted'],
                                       output_dict=True, digits=4)
        df_report = pd.DataFrame(report).T
        print(df_report.round(4))
        df_report.to_csv(str(LOGS / 'classification_report.csv'))
        """
    ),
    md("## 3.7 — Ensemble: average softmax across the best models, collapse to binary"),
    code(
        """
        frozen_models = sorted(CKPT.glob('*_best.keras'))
        frozen_models = [m for m in frozen_models if 'FineTuned' not in m.name]
        print(f'Found {len(frozen_models)} frozen-base checkpoints:')
        for m in frozen_models:
            print(f'  {m.name}')

        # Test set at 299x299 — matches the largest input size in the zoo.
        test_ds_ens = tf.keras.utils.image_dataset_from_directory(
            str(DATA / 'test'), image_size=(299, 299), batch_size=32,
            label_mode='int', class_names=CLASS_NAMES, shuffle=False)
        y_true_mc_ens  = np.concatenate([y.numpy() for _, y in test_ds_ens])
        y_true_bin_ens = RESTRICTED_MASK[y_true_mc_ens].astype(int)

        model_softmax = {}
        model_bin_acc = {}

        for model_path in frozen_models:
            name = model_path.stem.replace('_best', '')
            try:
                m = tf.keras.models.load_model(str(model_path))
                preds = m.predict(test_ds_ens, verbose=0)
                bin_prob = preds[:, RESTRICTED_IDX].sum(axis=1)
                acc = float(np.mean((bin_prob > 0.5).astype(int) == y_true_bin_ens))
                model_softmax[name] = preds
                model_bin_acc[name] = acc
                print(f'  {name}: bin_acc={acc:.4f}')
                del m; tf.keras.backend.clear_session()
            except Exception as e:
                print(f'  {name}: failed — {e}')

        ft_path = CKPT / 'FineTuned_best.keras'
        if ft_path.exists():
            ft_model = tf.keras.models.load_model(str(ft_path))
            ft_preds = ft_model.predict(test_ds_ens, verbose=0)
            ft_bin = ft_preds[:, RESTRICTED_IDX].sum(axis=1)
            ft_acc = float(np.mean((ft_bin > 0.5).astype(int) == y_true_bin_ens))
            model_softmax['FineTuned'] = ft_preds
            model_bin_acc['FineTuned'] = ft_acc
            print(f'  FineTuned: bin_acc={ft_acc:.4f}')
            del ft_model; tf.keras.backend.clear_session()

        ranked = sorted(model_bin_acc.items(), key=lambda x: -x[1])
        print('\\nRanking (binary accuracy):')
        for i, (n, s) in enumerate(ranked, 1):
            print(f'  #{i} {n}: {s:.4f}')

        top3 = [n for n, _ in ranked[:3]]
        all_n = [n for n, _ in ranked]

        ensembles = {}
        ensembles['Top3_Average']  = np.mean([model_softmax[n] for n in top3], axis=0)
        ensembles['All_Average']   = np.mean([model_softmax[n] for n in all_n], axis=0)
        w = np.array([model_bin_acc[n] for n in top3]); w = w / w.sum()
        ensembles['Top3_Weighted'] = np.average([model_softmax[n] for n in top3], axis=0, weights=w)

        print(f'\\n{"="*60}\\nENSEMBLE RESULTS (binary view)\\n{"="*60}')
        ensemble_results = []
        for ens_name, ens_softmax in ensembles.items():
            ens_bin = ens_softmax[:, RESTRICTED_IDX].sum(axis=1)
            y_ens   = (ens_bin > 0.5).astype(int)
            cm = confusion_matrix(y_true_bin_ens, y_ens, labels=[0, 1])
            TP, TN, FP, FN = cm[1,1], cm[0,0], cm[0,1], cm[1,0]
            acc  = (TP+TN) / max(TP+TN+FP+FN, 1)
            prec = TP / max(TP+FP, 1)
            rec  = TP / max(TP+FN, 1)
            f1   = 2*TP / max(2*TP+FP+FN, 1)
            mcc  = matthews_corrcoef(y_true_bin_ens, y_ens)
            ens_auc = roc_auc_score(y_true_bin_ens, ens_bin)
            ensemble_results.append({'ensemble': ens_name,
                                     'models': ', '.join(top3) if '3' in ens_name else 'all',
                                     'accuracy': acc, 'precision': prec, 'recall': rec,
                                     'f1': f1, 'mcc': mcc, 'auc': ens_auc})
            print(f'{ens_name}: Acc={acc:.4f}  P={prec:.4f}  R={rec:.4f}  '
                  f'F1={f1:.4f}  MCC={mcc:.4f}  AUC={ens_auc:.4f}')

        best_single = ranked[0]
        best_ens = max(ensemble_results, key=lambda x: x['f1'])
        print(f'\\nBest single model: {best_single[0]} (bin_acc={best_single[1]:.4f})')
        print(f'Best ensemble:     {best_ens["ensemble"]} (F1={best_ens["f1"]:.4f})')

        pd.DataFrame(ensemble_results).to_csv(str(INFER / 'ensemble_results.csv'), index=False)
        print('\\nNotebook 3 complete. Open Notebook 4 and Run all.')
        """
    ),
]
write_notebook("3_FineTuning_Colab.ipynb", nb3_cells)


# ─────────────────────────────────────────────────────────────────────────────
# Notebook 4 — Grad-CAM and failure analysis
# ─────────────────────────────────────────────────────────────────────────────
nb4_cells: list[dict] = [
    md(
        """
        # Notebook 4 — Interpretability: Grad-CAM and Failure Analysis

        Grad-CAM heatmaps for one image per breed, plus a gallery of the most
        confident binary misclassifications.

        Because the model is now a 36-way softmax, the "restricted" probability
        is the sum of restricted-class softmax outputs. Grad-CAM is computed
        against this collapsed scalar so the heatmap shows what pushed the
        model toward *restricted* (vs unrestricted) — not just toward one breed.
        """
    ),
    *PREAMBLE,
    MANIFEST_LOAD,
    md("## 4.1 — Load the best model"),
    code(
        """
        import os, glob, random, cv2
        from collections import defaultdict
        import matplotlib.pyplot as plt

        model_path = str(CKPT / 'FineTuned_best.keras')
        if not os.path.exists(model_path):
            model_path = str(CKPT / 'InceptionResNetV2_best.keras')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'No model found in {CKPT}')

        model = tf.keras.models.load_model(model_path)
        print(f'Loaded model: {os.path.basename(model_path)}')

        base_model = None
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model):
                base_model = layer
                break

        conv_layers = [l.name for l in base_model.layers if 'conv' in l.name]
        LAST_CONV = conv_layers[-1]
        print(f'Base: {base_model.name}  | Last conv: {LAST_CONV}')

        GRADCAM_DIR = INFER / 'gradcam_images'
        GRADCAM_DIR.mkdir(parents=True, exist_ok=True)
        """
    ),
    md("## 4.2 — Grad-CAM function (multi-class softmax → restricted scalar)"),
    code(
        """
        # The saved model already contains its preprocess_input layer, so feed
        # raw 0–255 images. Split the model into pre/post layers around the
        # nested base_model so we can tap the last conv activations.
        RESTRICTED_IDX_TF = tf.constant(RESTRICTED_IDX, dtype=tf.int32)
        BASE_IDX    = next(i for i, l in enumerate(model.layers) if l is base_model)
        PRE_LAYERS  = model.layers[1:BASE_IDX]        # skip outer InputLayer
        POST_LAYERS = model.layers[BASE_IDX+1:]
        TAP_MODEL   = tf.keras.Model(
            inputs=base_model.input,
            outputs=[base_model.get_layer(LAST_CONV).output, base_model.output])

        def run_gradcam(model, img_path, target_size=(299, 299), alpha=0.45):
            img = tf.keras.utils.load_img(img_path, target_size=target_size)
            raw = tf.keras.utils.img_to_array(img)                 # 0–255 float
            img_array = np.expand_dims(raw, 0)

            probs = model.predict(img_array, verbose=0)[0]
            p_restricted = float(probs[RESTRICTED_IDX].sum())
            label = 'RESTRICTED' if p_restricted > 0.5 else 'UNRESTRICTED'
            conf  = p_restricted if p_restricted > 0.5 else 1 - p_restricted
            top_breed = CLASS_NAMES[int(probs.argmax())]

            with tf.GradientTape() as tape:
                x = tf.cast(img_array, tf.float32)
                for layer in PRE_LAYERS:
                    x = layer(x)
                conv_out, base_out = TAP_MODEL(x)
                tape.watch(conv_out)
                z = base_out
                for layer in POST_LAYERS:
                    z = layer(z)
                p_restricted_t = tf.reduce_sum(tf.gather(z[0], RESTRICTED_IDX_TF))
            grads  = tape.gradient(p_restricted_t, conv_out)
            pooled = tf.reduce_mean(grads, axis=(0, 1, 2))

            heatmap = tf.reduce_sum(conv_out[0] * pooled, axis=-1)
            heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
            heatmap = heatmap.numpy()

            heatmap_resized = cv2.resize(heatmap, target_size)
            heatmap_colour  = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
            original = raw.astype(np.uint8)
            overlay  = cv2.addWeighted(heatmap_colour, alpha, original, 1 - alpha, 0)

            breed_prefix = '_'.join(os.path.basename(img_path).split('_')[:-1])
            cat = 'restricted' if breed_prefix in RESTRICTED_SET else 'unrestricted'
            out_path = str(GRADCAM_DIR / f'{cat}_{breed_prefix}_gradcam.jpg')

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(original / 255.0)
            axes[0].set_title(f'Original\\n{label} ({conf:.1%})  top={top_breed}')
            axes[1].imshow(heatmap_colour[..., ::-1] / 255.0)
            axes[1].set_title('Grad-CAM (restricted-class)')
            axes[2].imshow(overlay[..., ::-1] / 255.0)
            axes[2].set_title('Overlay')
            for a in axes: a.axis('off')
            plt.tight_layout()
            fig.savefig(out_path, dpi=150, bbox_inches='tight')
            plt.show(); plt.close()
            print(f'  saved {out_path}')
            return p_restricted, label, conf, top_breed

        print('Grad-CAM function ready.')
        """
    ),
    md("## 4.3 — Run Grad-CAM on one image per breed"),
    code(
        """
        def pick_one_per_breed(base_dir):
            paths = []
            for breed in sorted(os.listdir(base_dir)):
                bdir = os.path.join(base_dir, breed)
                if not os.path.isdir(bdir): continue
                imgs = glob.glob(os.path.join(bdir, '*.jpg'))
                if imgs:
                    paths.append(random.choice(imgs))
            print(f'  Found {len(paths)} breeds in {os.path.basename(base_dir)}')
            return paths

        random.seed(58)
        all_imgs = pick_one_per_breed(str(DATA / 'test'))
        print(f'\\nRunning Grad-CAM on {len(all_imgs)} images...\\n')

        for i, path in enumerate(all_imgs, 1):
            print(f'[{i}/{len(all_imgs)}] {os.path.basename(path)}')
            run_gradcam(model, path)

        print(f'\\nGrad-CAM complete. Triplets in: {GRADCAM_DIR}')
        """
    ),
    md("## 4.4 — Failure analysis (binary view)"),
    code(
        """
        test_ds = tf.keras.utils.image_dataset_from_directory(
            str(DATA / 'test'), image_size=(299, 299), batch_size=32,
            label_mode='int', class_names=CLASS_NAMES, shuffle=False)

        file_paths = test_ds.file_paths
        y_true_mc  = np.concatenate([y.numpy() for _, y in test_ds])
        y_prob_mc  = model.predict(test_ds).reshape(-1, NUM_CLASSES)
        y_prob_bin = y_prob_mc[:, RESTRICTED_IDX].sum(axis=1)
        y_true_bin = RESTRICTED_MASK[y_true_mc].astype(int)
        y_pred_bin = (y_prob_bin > 0.5).astype(int)

        wrong_idx = np.where(y_pred_bin != y_true_bin)[0]

        if len(wrong_idx) > 0:
            confidence   = np.abs(y_prob_bin[wrong_idx] - 0.5)
            sorted_wrong = wrong_idx[np.argsort(-confidence)]

            n_show = min(8, len(sorted_wrong))
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            axes = axes.flatten()

            for i, idx in enumerate(sorted_wrong[:n_show]):
                img = plt.imread(file_paths[idx])
                true_label = 'Restricted' if y_true_bin[idx] == 1 else 'Unrestricted'
                pred_label = 'Restricted' if y_pred_bin[idx] == 1 else 'Unrestricted'
                true_breed = CLASS_NAMES[y_true_mc[idx]]
                pred_breed = CLASS_NAMES[int(y_prob_mc[idx].argmax())]
                axes[i].imshow(img)
                axes[i].set_title(
                    f'True: {true_label} ({true_breed})\\n'
                    f'Pred: {pred_label} ({pred_breed}, p={y_prob_bin[idx]:.2f})',
                    fontsize=9, color='red')
                axes[i].axis('off')

            for i in range(n_show, len(axes)): axes[i].set_visible(False)
            plt.suptitle('Most Confident Binary Misclassifications',
                         fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(str(INFER / 'failure_gallery.png'), dpi=300, facecolor='white')
            plt.show()
            print(f'Failure gallery saved.')
        else:
            print('No misclassifications on the test set.')

        print(f'\\nNotebook 4 complete.')
        print(f'  Grad-CAM images: {GRADCAM_DIR}')
        print(f'  Failure gallery: {INFER / "failure_gallery.png"}')
        """
    ),
]
write_notebook("4_GradCAM_Colab.ipynb", nb4_cells)

print("\nDone.")
