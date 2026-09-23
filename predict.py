"""Predict shot type and camera angle for one or more images.

    python predict.py still.jpg another.png
"""
import argparse
import os

import joblib
from PIL import Image

from common import HEADS, MODEL_DIR, ClipEmbedder


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('images', nargs='+')
    parser.add_argument('--top', type=int, default=3, help='how many guesses to show per head')
    args = parser.parse_args()

    models = {head: joblib.load(os.path.join(MODEL_DIR, f'{head}.joblib'))
              for head in HEADS if os.path.exists(os.path.join(MODEL_DIR, f'{head}.joblib'))}
    if not models:
        raise SystemExit(f'No trained models in {MODEL_DIR}/. Run train.py first.')
    with open(os.path.join(MODEL_DIR, 'clip_model.txt')) as f:
        embedder = ClipEmbedder(f.read().strip())

    vectors = embedder.embed([Image.open(path).convert('RGB') for path in args.images])
    for path, vector in zip(args.images, vectors):
        print(path)
        for head, model in models.items():
            probs = model.predict_proba([vector])[0]
            ranked = sorted(zip(model.classes_, probs), key=lambda p: -p[1])[:args.top]
            print(f'  {head}: ' + ', '.join(f'{label} {p:.0%}' for label, p in ranked))


if __name__ == '__main__':
    main()
