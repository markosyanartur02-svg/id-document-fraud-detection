"""Sanity check: can file metadata alone predict fraud on an unseen document type?

If fraudulent images were saved at a different size or JPEG quality, a CNN could
score well without learning anything about forgery. This trains a gradient-boosted
model on metadata only (image size, file size, bytes per pixel, JPEG quantisation
tables) and evaluates it on each held-out document type. An AUC near 0.5 means
the metadata carries no usable fraud signal.

Usage: python shortcut_check.py [labels_csv] [image_dir]
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import pandas as pd
from PIL import Image
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

FEATURES = ['width', 'height', 'file_size', 'bytes_per_pixel', 'qtable_luma', 'qtable_chroma']


def file_metadata(image_path, img_dir):
    path = os.path.join(img_dir, image_path.split('/')[-1])
    with Image.open(path) as img:
        width, height = img.size
        qtables = getattr(img, 'quantization', None) or {}
    return {
        'width': width,
        'height': height,
        'file_size': os.path.getsize(path),
        'qtable_luma': sum(qtables.get(0, [])),
        'qtable_chroma': sum(qtables.get(1, [])),
    }


def main(labels_path, img_dir):
    labels = pd.read_csv(labels_path)
    with ProcessPoolExecutor() as ex:
        meta = pd.DataFrame(ex.map(partial(file_metadata, img_dir=img_dir), labels['image_path'], chunksize=500))
    meta['bytes_per_pixel'] = meta['file_size'] / (meta['width'] * meta['height'])
    df = pd.concat([labels, meta], axis=1)

    print('Metadata-only ROC-AUC on each held-out document type:')
    for held_out in sorted(df['type'].unique()):
        seen, unseen = df[df['type'] != held_out], df[df['type'] == held_out]
        clf = HistGradientBoostingClassifier(random_state=42).fit(seen[FEATURES], seen['label'])
        auc = roc_auc_score(unseen['label'], clf.predict_proba(unseen[FEATURES])[:, 1])
        print(f'  {held_out:15s} {auc:.3f}')


if __name__ == '__main__':
    main(*(sys.argv[1:] or ['data/train_labels.csv', 'train']))
