# Neural Networks from Scratch

Implementing neural network architectures from scratch using only NumPy.

## Implemented

- **Binary Classifier** — Fully connected MLP with backpropagation, BCE loss, and train/val/test loop.

## In Progress

- **Transformer**

## Structure

```
activation_functions.py   # ReLU, Sigmoid (classes with forward/backward)
losses.py                 # BCELoss (classes with forward/backward)
binary_classifier.py      # Layers, Model, Train
```

## Sample Output

```
Epoch   0 | Train Loss: 0.5411 | Validation Loss: 1.0952
Epoch  50 | Train Loss: 0.0488 | Validation Loss: 0.0017
Epoch  90 | Train Loss: 0.0428 | Validation Loss: 0.0014
-----------------------------------
Train Loss: 0.0364
Validation Loss: 0.0012
Test Loss: 0.0016
-----------------------------------
```

## Requirements

```
numpy
```