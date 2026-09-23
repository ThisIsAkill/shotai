"""Compute CLIP embeddings for every image in labels.csv and cache them.

Only images missing from datasets/embeddings.npz are processed, so re-running
after scraping more tags is cheap.

    python embed.py
    python embed.py --model openai/clip-vit-large-patch14   # slower, more accurate
"""
import argparse
import os

import numpy as np
from PIL import Image

from common import CLIP_MODEL, EMBEDDINGS_NPZ, IMAGE_DIR, ClipEmbedder, read_labels


def load_cache(model_name):
    if not os.path.exists(EMBEDDINGS_NPZ):
        return {}
    cache = np.load(EMBEDDINGS_NPZ)
    if str(cache['model']) != model_name:
        print(f'Cache was built with {cache["model"]}, recomputing for {model_name}')
        return {}
    return dict(zip(cache['ids'], cache['vectors']))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', default=CLIP_MODEL, help='Hugging Face CLIP model name')
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()

    rows = read_labels()
    cache = load_cache(args.model)
    todo = [row for id_, row in rows.items() if id_ not in cache]
    print(f'{len(rows)} image(s), {len(todo)} to embed')
    if not todo:
        return

    embedder = ClipEmbedder(args.model)
    for start in range(0, len(todo), args.batch_size):
        batch, images = [], []
        for row in todo[start:start + args.batch_size]:
            try:
                images.append(Image.open(os.path.join(IMAGE_DIR, row['file'])).convert('RGB'))
                batch.append(row['id'])
            except OSError as e:
                print(f'Skipping {row["file"]}: {e}')
        if images:
            cache.update(zip(batch, embedder.embed(images)))
        print(f'{min(start + args.batch_size, len(todo))}/{len(todo)}')

    ids = sorted(cache, key=int)
    np.savez(EMBEDDINGS_NPZ, ids=np.array(ids), vectors=np.stack([cache[i] for i in ids]),
             model=np.array(args.model))
    print(f'Saved {len(ids)} embedding(s) to {EMBEDDINGS_NPZ}')


if __name__ == '__main__':
    main()
