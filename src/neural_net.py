"""
Feed-forward neural network (multi-layer perceptron) in PyTorch, trained by us.

Architecture:  input -> [Linear -> activation -> (dropout)] x len(hidden) -> Linear -> class scores
Training:      mini-batch gradient descent on the cross-entropy loss.
               Each epoch = one pass over the training data in shuffled batches.

Hyperparameters exposed (all explored in the notebook):
  hidden        tuple of hidden-layer sizes, e.g. (256, 64). () = no hidden layer,
                which is plain multinomial logistic regression (a linear model)
  activation    "relu", "gelu", "tanh", "sigmoid"
  dropout       probability of zeroing each hidden unit during training (regularisation)
  lr            learning rate (step size)
  optimizer     "adam" or "sgd" (with momentum 0.9)
  weight_decay  L2 penalty on the weights (regularisation)
  batch_size    samples per gradient step
  epochs        passes over the training data
We record train/validation loss and accuracy after every epoch, and keep the
weights from the epoch with the best validation accuracy (early stopping).
"""
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid}


class MLP(nn.Module):
    def __init__(self, in_dim, n_classes, hidden=(256, 64), activation="relu", dropout=0.0):
        super().__init__()
        layers, prev = [], in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), ACTIVATIONS[activation]()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, n_classes))   # raw scores ("logits"); softmax is inside the loss
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _make_optimizer(model, name, lr, weight_decay):
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(name)


@torch.no_grad()
def _evaluate(model, X, y, loss_fn, batch=4096):
    model.eval()
    losses, correct = 0.0, 0
    for i in range(0, len(X), batch):
        out = model(X[i:i + batch])
        losses += loss_fn(out, y[i:i + batch]).item() * len(out)
        correct += (out.argmax(1) == y[i:i + batch]).sum().item()
    return losses / len(X), correct / len(X)


def train_mlp(X_train, y_train, X_val=None, y_val=None, n_classes=None,
              hidden=(256, 64), activation="relu", dropout=0.0,
              lr=1e-3, optimizer="adam", weight_decay=0.0,
              batch_size=128, epochs=50, seed=42, restore_best=True, patience=None,
              verbose=False):
    """
    Train an MLP. Returns (model, history DataFrame, best_epoch).
    If X_val is None (final model on all training data) we simply train for `epochs`.
    patience: early stopping - stop if validation accuracy has not improved for this
              many epochs (None = always train for `epochs`).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    n_classes = n_classes or int(np.max(y_train)) + 1
    Xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.long)
    has_val = X_val is not None
    if has_val:
        Xv = torch.tensor(X_val, dtype=torch.float32)
        yv = torch.tensor(y_val, dtype=torch.long)

    model = MLP(Xt.shape[1], n_classes, hidden, activation, dropout)
    opt = _make_optimizer(model, optimizer, lr, weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))

    rows, best_acc, best_epoch, best_state = [], -1.0, 0, None
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)   # forward pass + loss
            loss.backward()                 # backpropagation: gradients of loss w.r.t. weights
            opt.step()                      # update the weights
        tr_loss, tr_acc = _evaluate(model, Xt, yt, loss_fn)
        row = dict(epoch=epoch, train_loss=tr_loss, train_acc=tr_acc)
        if has_val:
            va_loss, va_acc = _evaluate(model, Xv, yv, loss_fn)
            row.update(val_loss=va_loss, val_acc=va_acc)
            if va_acc > best_acc:
                best_acc, best_epoch = va_acc, epoch
                best_state = copy.deepcopy(model.state_dict())
        rows.append(row)
        if has_val and patience is not None and epoch - best_epoch >= patience:
            break
        if verbose and (epoch % 10 == 0 or epoch == 1):
            print({k: round(v, 4) for k, v in row.items()})

    if has_val and restore_best and best_state is not None:
        model.load_state_dict(best_state)
    if not has_val:
        best_epoch = epochs
    return model, pd.DataFrame(rows), best_epoch


@torch.no_grad()
def predict_proba(model, X, batch=4096):
    model.eval()
    X = torch.tensor(X, dtype=torch.float32)
    out = [torch.softmax(model(X[i:i + batch]), dim=1) for i in range(0, len(X), batch)]
    return torch.cat(out).numpy()
