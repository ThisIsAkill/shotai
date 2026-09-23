"""Shared config and CLIP helpers for the ShotAI pipeline."""
import csv
import os

import numpy as np

DATA_DIR = 'datasets'
IMAGE_DIR = os.path.join(DATA_DIR, 'images')
LABELS_CSV = os.path.join(DATA_DIR, 'labels.csv')
EMBEDDINGS_NPZ = os.path.join(DATA_DIR, 'embeddings.npz')
MODEL_DIR = 'models'

CLIP_MODEL = 'openai/clip-vit-base-patch32'

# Each head is trained on images carrying exactly one of its tags.
# Tag names are exactly as shot.cafe spells them.
HEADS = {
    'shot_type': [
        'extreme close-up',
        'close-up',
        'medium close-up',
        'medium shot',
        'medium wide',
        'wide shot',
        'extreme wide',
    ],
    'angle': [
        'low-angle',
        'high-angle',
        'overhead',
        'dutch-angle',
    ],
}

# shot.cafe tags that mean the same thing as one of the classes above.
ALIASES = {
    'medium': 'medium shot',
    'wide': 'wide shot',
    'top-down': 'overhead',
}

ALL_TAGS = [tag for tags in HEADS.values() for tag in tags] + list(ALIASES)


def read_labels(path=LABELS_CSV):
    """Return {id: {'id', 'file', 'film', 'tags': set}} from labels.csv."""
    rows = {}
    if not os.path.exists(path):
        return rows
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            row['tags'] = set(filter(None, row['tags'].split(';')))
            rows[row['id']] = row
    return rows


def write_labels(rows, path=LABELS_CSV):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'file', 'film', 'tags'])
        writer.writeheader()
        for row in sorted(rows.values(), key=lambda r: int(r['id'])):
            writer.writerow({**row, 'tags': ';'.join(sorted(row['tags']))})
    os.replace(tmp, path)


def head_label(tags, head):
    """The single class of `head` among `tags`, or None if zero or several match."""
    tags = {ALIASES.get(t, t) for t in tags}
    matches = tags & set(HEADS[head])
    return matches.pop() if len(matches) == 1 else None


class ClipEmbedder:
    def __init__(self, model_name=CLIP_MODEL):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def embed(self, images):
        """L2-normalised embeddings for a list of PIL images."""
        inputs = self.processor(images=images, return_tensors='pt').to(self.device)
        with self.torch.no_grad():
            features = self.model.get_image_features(**inputs)
        # transformers >= 5 wraps the projected features in an output object.
        features = getattr(features, 'pooler_output', features)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().astype(np.float32)
