"""Small, readable PyTorch building blocks, independent of the GUI."""
import torch
from torch import nn


def build_mlp(description: str, num_classes: int) -> nn.Sequential:
    """Parse hidden layers and append the mandatory classification layer.

    Inputs are two coordinates. Outputs are raw scores (logits), one per class.
    CrossEntropyLoss applies log-softmax internally, so no output sigmoid or
    softmax is needed. The word 'logistic' means sigmoid in a hidden layer.
    """
    if not 2 <= num_classes <= 5:
        raise ValueError("Choose between 2 and 5 classes.")
    layers = []
    input_size = 2
    lines = description.splitlines()
    if len(lines) > 40:
        raise ValueError("Use at most 40 lines for this interactive demo.")
    for line_number, line in enumerate(lines, start=1):
        words = line.split("#", 1)[0].lower().split()
        if not words:
            continue
        name, *args = words
        try:
            if name == "linear" and len(args) == 1:
                output_size = int(args[0])
                if not 1 <= output_size <= 512:
                    raise ValueError("Layer width must be between 1 and 512.")
                layers.append(nn.Linear(input_size, output_size))
                input_size = output_size
            elif name == "relu" and not args:
                layers.append(nn.ReLU())
            elif name in ("logistic", "sigmoid") and not args:
                layers.append(nn.Sigmoid())
            elif name == "tanh" and not args:
                layers.append(nn.Tanh())
            elif name == "dropout" and len(args) == 1:
                probability = float(args[0])
                if not 0 <= probability < 1:
                    raise ValueError("Dropout probability must be in [0, 1).")
                layers.append(nn.Dropout(probability))
            else:
                raise ValueError("Expected linear N, relu, logistic, tanh or dropout P.")
        except ValueError as error:
            raise ValueError(f"Line {line_number}: {error}") from error
    layers.append(nn.Linear(input_size, num_classes))
    return nn.Sequential(*layers)


def make_optimizer(model, name: str, learning_rate: float):
    optimizers = {"Adam": torch.optim.Adam, "SGD": torch.optim.SGD,
                  "RMSprop": torch.optim.RMSprop, "AdamW": torch.optim.AdamW}
    return optimizers[name](model.parameters(), lr=learning_rate)


def train_one_epoch(model, optimizer, features, labels):
    """One full-batch epoch: all drawn points contribute to one update."""
    model.train()                     # Enable dropout during training.
    optimizer.zero_grad()             # Clear gradients from the previous step.
    logits = model(features)          # Forward pass: [number of points, classes].
    loss = nn.functional.cross_entropy(logits, labels)
    if not torch.isfinite(loss):
        raise ValueError("Loss is not finite. Reduce the learning rate and reset the model.")
    loss.backward()                   # Compute gradients using backpropagation.
    optimizer.step()                   # Update weights and biases.
    return loss.item()


@torch.no_grad()
def predict_classes(model, features):
    model.eval()                      # Disable dropout for a stable visualization.
    return model(features).argmax(dim=1)


@torch.no_grad()
def evaluate(model, features, labels):
    model.eval()
    logits = model(features)
    loss = nn.functional.cross_entropy(logits, labels).item()
    accuracy = (logits.argmax(dim=1) == labels).float().mean().item()
    return loss, accuracy
