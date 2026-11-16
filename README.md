# Computer Vision from the ground up

Code for a YouTube tutorial series on deep learning and computer vision in PyTorch.

Every part is one self-contained script. The series starts with a small MLP on MNIST and gets a bit harder with each step. The goal at the end is a Mini-ViT written from scratch.

Videos: TODO link

## Idea

Each script adds one or two new things on top of the previous one, and nothing else. If you go through them in order, you never have to deal with a dataset, an architecture and a training trick that are all new at the same time.

There is no shared utils module on purpose. Each file contains everything it needs: data loading, model, training loop, evaluation. That means some code repeats, but you can open any single file and read it from top to bottom.

## Part 1: Classification

| # | File | Dataset | What is new |
|---|------|---------|-------------|
| 01 | `classification_01_MNIST.py` | MNIST | MLP, training loop, evaluation loop, cosine LR schedule |
| 02 | `classification_02_FMNIST.py` | Fashion-MNIST | first CNN: convolutions, max pooling, dropout |
| 03 | `classification_03_CIFAR10.py` | CIFAR-10 | color images, conv blocks with BatchNorm, data augmentation, global average pooling, weight decay |
| 04 | `classification_04_SVHN.py` | SVHN | real-world photos, Dropout2d, OneCycleLR stepped per batch |
| 05 | `classification_05_GTSRB.py` | GTSRB | 43 classes, images of different sizes, class imbalance handled with a weighted loss |

## Roadmap

Planned, not in the repo yet. Order and scope may still change.

- Residual connections and a small ResNet
- Transfer learning and finetuning of a pretrained backbone
- Beyond classification: detection and segmentation
- Attention and self-attention on image patches
- Mini-ViT from scratch: patch embedding, positional embedding, transformer encoder, classification head

## Setup

You need Python 3 and PyTorch with torchvision. SVHN additionally needs scipy, because torchvision reads the dataset from `.mat` files.

```bash
pip install torch torchvision scipy
```

For a CUDA build of PyTorch, use the install command from [pytorch.org](https://pytorch.org/get-started/locally/).

## Running

```bash
python classification_01_MNIST.py
```

The datasets are downloaded automatically on the first run into `./data`, relative to the directory you start the script from. The scripts use the GPU if one is available and fall back to the CPU otherwise. Parts 01 and 02 are fine on a CPU, from 03 on a GPU saves a lot of time.

Loss and accuracy for train and test are printed once per epoch.

## Notes

- Hyperparameters are constants at the top of each file or right above the training loop. Change them and see what happens, that is the point.
- The scripts are written to be read, not to reach the best possible accuracy.
- The test split is used for evaluation after every epoch. For a tutorial that is fine, for real work you would hold out a separate validation set.
