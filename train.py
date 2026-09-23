"""Train one classifier per head (shot type, camera angle) on cached CLIP embeddings.

The test set is split by film, so frames from one movie never appear in both
training and test, which would inflate accuracy. After reporting test scores
each model is refit on all data and saved to models/<head>.joblib.

    python train.py
"""
import argparse
import os
from collections import Counter

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold

from common import EMBEDDINGS_NPZ, HEADS, MODEL_DIR, head_label, read_labels


def train_head(head, rows, vectors, min_per_class, C):
    samples = [(id_, head_label(row['tags'], head), row['film'])
               for id_, row in rows.items() if id_ in vectors]
    samples = [s for s in samples if s[1] is not None]
    counts = Counter(label for _, label, _ in samples)
    kept = {label for label, n in counts.items() if n >= min_per_class}
    dropped = {label: n for label, n in counts.items() if label not in kept}
    if dropped:
        print(f'Skipping classes with < {min_per_class} images: {dropped}')
    samples = [s for s in samples if s[1] in kept]
    if len(kept) < 2:
        print(f'Not enough labelled data for {head}; scrape more tags first.')
        return

    X = np.stack([vectors[id_] for id_, _, _ in samples])
    y = np.array([label for _, label, _ in samples])
    films = np.array([film for _, _, film in samples])
    print(f'{len(y)} images from {len(set(films))} films')

    def make_model():
        return LogisticRegression(C=C, max_iter=5000, class_weight='balanced')

    # Hold out ~20% of films for testing.
    train_idx, test_idx = next(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
                               .split(X, y, groups=films))
    model = make_model().fit(X[train_idx], y[train_idx])
    y_pred = model.predict(X[test_idx])
    print(classification_report(y[test_idx], y_pred, zero_division=0))
    print('Confusion matrix (rows = true, columns = predicted):')
    print('  ' + ', '.join(model.classes_))
    print(confusion_matrix(y[test_idx], y_pred, labels=model.classes_))

    final = make_model().fit(X, y)
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, f'{head}.joblib')
    joblib.dump(final, path)
    print(f'Saved {path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--min-per-class', type=int, default=20)
    parser.add_argument('--C', type=float, default=10.0, help='inverse regularisation strength')
    args = parser.parse_args()

    cache = np.load(EMBEDDINGS_NPZ)
    vectors = dict(zip(cache['ids'], cache['vectors']))
    rows = read_labels()
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, 'clip_model.txt'), 'w') as f:
        f.write(str(cache['model']))

    for head in HEADS:
        print(f'\n=== {head} ===')
        train_head(head, rows, vectors, args.min_per_class, args.C)


if __name__ == '__main__':
    main()
