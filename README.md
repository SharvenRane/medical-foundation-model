# medical-foundation-model

A compact, end to end example of building a small domain foundation encoder for
medical images with self supervision, then checking what it learned by linear
probing. The whole thing is designed to run on a laptop CPU in seconds to a
couple of minutes, with MedMNIST as the unlabeled image source.

The idea mirrors how larger medical foundation models are built. You pretrain an
image encoder on lots of unlabeled scans using a self supervised objective, you
throw away the pretraining head, and you keep the backbone. Then you measure the
quality of the backbone by freezing it and fitting a simple linear classifier on
a small labeled set. If the frozen features carry real structure, that linear
probe beats the same probe on a randomly initialised encoder.

## What is inside

The encoder is a small convolutional backbone. The self supervised method is
SimCLR: for each image you create two randomly augmented views, push both
through the encoder and a projection head, and minimise the NT-Xent contrastive
loss so matching views attract and everything else repels. After pretraining the
projection head is discarded and the backbone embeddings feed a logistic
regression probe.

Source layout:

- `src/encoder.py` the convolutional backbone, the projection head, and the
  combined SimCLR module.
- `src/augment.py` the stochastic two view augmentation pipeline (random resized
  crop, horizontal flip, brightness and contrast jitter, Gaussian noise).
- `src/loss.py` the NT-Xent contrastive loss.
- `src/data.py` a MedMNIST loader plus a synthetic dataset used by the fast
  offline tests.
- `src/pretrain.py` the SimCLR training loop, returning the trained model and the
  per epoch loss history.
- `src/probe.py` feature extraction and the frozen encoder linear probe, plus a
  helper that builds a matching random encoder baseline.
- `src/run_demo.py` an end to end script that pretrains on MedMNIST and prints the
  probe comparison.

## Setup

```
pip install -r requirements.txt
```

PyTorch, scikit-learn, and MedMNIST do the heavy lifting. Everything runs on CPU.

## Running the demo

```
python -m src.run_demo --flag pathmnist --pretrain-size 1500 \
    --probe-train-size 400 --probe-test-size 400 --epochs 6 --batch-size 64
```

The first run downloads the chosen MedMNIST dataset (PathMNIST here, a colorectal
histology dataset with nine tissue classes). Later runs read the local cache.

## A real run

The numbers below come from one actual CPU run of the command above on
PathMNIST. They are not tuned and will shift a little with the seed and the
machine, but the shape of the result is stable: the contrastive loss falls
across epochs, and the frozen pretrained features beat the random encoder under
the linear probe.

```
Pretraining loss per epoch:
  epoch 1: 3.9886
  epoch 2: 3.6357
  epoch 3: 3.5889
  epoch 4: 3.5115
  epoch 5: 3.4852
  epoch 6: 3.4563
Linear probe accuracy (pretrained encoder): 0.6975
Linear probe accuracy (random encoder):     0.6125
Improvement: +0.0850
```

A short pretraining schedule already lifts probe accuracy well above the random
baseline and far above the nine class chance level near 0.11. Longer schedules
and more pretraining images widen the gap further.

## Tests

```
pytest tests/ -q
```

The tests are behaviour checks, not snapshots of exact numbers:

- The encoder produces the expected embedding shapes for grayscale and RGB input
  and stays differentiable.
- The NT-Xent loss is a finite scalar, drops when the two views are aligned
  versus mismatched, and rejects shape mismatches.
- The augmentation pipeline keeps shape and value range and yields two views that
  actually differ.
- Pretraining drives the contrastive loss down over epochs on a small synthetic
  dataset.
- On real MedMNIST images the pretrained encoder beats a random encoder under the
  linear probe. Randomly initialised CNNs are strong feature extractors on clean
  synthetic textures, so this honest gap is measured on real medical data. The
  test skips itself when the dataset is not already cached locally, which keeps
  the suite offline by default.

## Notes and honest limits

This is a teaching scale model, not a production foundation encoder. The backbone
is tiny, the images are 28 by 28, and the pretraining schedule is short. The
point is to show the full loop working with correct, tested components:
contrastive pretraining that learns, and a frozen feature probe that proves the
learning was useful. Scaling up means a larger backbone, larger images, longer
training, and a stronger augmentation stack, with the same overall structure.
```
