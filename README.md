# ShotAI

Classifies film stills by **shot type** (close-up, wide shot, …) and **camera angle**
(low, high, overhead, Dutch), trained on tagged stills from [shot.cafe](https://shot.cafe).

Instead of training a CNN from scratch, each image is turned into a
[CLIP](https://huggingface.co/openai/clip-vit-base-patch32) embedding once. A small
logistic-regression classifier is then trained on those embeddings. Training takes seconds on a
CPU and needs only a few hundred images per class.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python scrape.py      # download stills + build datasets/labels.csv
python embed.py       # compute CLIP embeddings (cached; only new images are processed)
python train.py       # train + evaluate, saves models/shot_type.joblib and models/angle.joblib
python predict.py still.jpg
```

Example output:

```
still.jpg
  shot_type: wide shot 95%, close-up 5%
  angle: high-angle 79%, low-angle 17%, overhead 5%
```

For a quick trial, scrape a few pages of a few tags:

```bash
python scrape.py --tags close-up "wide shot" low-angle high-angle overhead --max-pages 4
```

With that setting (~390 images, 16 s to scrape, 30 s to embed on a CPU) the test set scored
92% on close-up vs. wide and 67% on high/low/overhead angles.

### Full dataset

`python scrape.py` with the default tags gives 16,159 stills from 190 films (5 GB, about
2,100 page requests). Embedding them takes ~19 min on a CPU and training ~30 s. Test set = held-out films:

| Head | Classes | Images | Accuracy |
|---|---|---|---|
| Shot type | 7 (extreme close-up … extreme wide) | 15,897 | 64% exact, 97% within one step |
| Angle | 4 (low, high, overhead, Dutch) | 1,609 | 76% (low 84% F1, high 76%, overhead 61%, Dutch 0%) |

Almost all shot-type errors are between neighbouring sizes (medium shot vs. medium wide,
wide vs. extreme wide), where tagging on shot.cafe is itself subjective. Dutch angle has only
59 examples and is never predicted correctly yet.

## How it works

- **`scrape.py`** downloads each still once to `datasets/images/<id>.<ext>`. When the same
  still appears under several tags, its tags are merged into one row of `datasets/labels.csv`
  (`id, file, film, tags`). Re-running it only downloads what is missing. It sends at most 4
  requests at a time (`--concurrency`).
- **`common.py`** defines which shot.cafe tags belong to each classifier (`HEADS`) and which
  tags are synonyms (`ALIASES`). Edit it to add classes. An image is used to train a
  classifier only if it has exactly one of that classifier's tags.
- **`train.py`** holds out ~20% of *films* (not images) for testing. Frames from the same movie
  look alike, so a random split would inflate the score. It prints a classification report
  and confusion matrix, then refits on all data.
- For a bigger, slower and more accurate model: `python embed.py --model openai/clip-vit-large-patch14`,
  then retrain.

## Limitations

- shot.cafe has no "eye-level" tag, so the angle classifier always picks one of
  low / high / overhead / Dutch, even for a neutral shot.
- Tags on shot.cafe are incomplete. A still without a `low-angle` tag isn't necessarily
  not low-angle, which is why untagged images are left out rather than treated as negatives.
- The stills are copyrighted film frames. Check shot.cafe's terms before scraping at scale,
  and don't redistribute the dataset.

## Tests

```bash
python -m pytest tests
```

## Legacy

`detection.py` is the original from-scratch Keras CNN. It does not run as-is: `ImageDataGenerator.flow()`
cannot take a dict of labels, Keras 3 rejects `metrics=['accuracy']` on a two-output model,
and the printed test accuracies are swapped.
