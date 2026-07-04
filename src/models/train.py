import copy

import torch
import torch.nn as nn
import torch.optim as optim


def create_optimizer(
    model: nn.Module,
    learning_rate: float = 1e-3,
):
    """
    Crea un optimizador Adam.
    """

    return optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
):
    """
    Entrena el modelo durante una época.
    """

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for features, labels in dataloader:

        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(features)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()

        total += labels.size(0)

    epoch_loss = running_loss / len(dataloader)

    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def evaluate(
    model,
    dataloader,
    criterion,
    device,
):
    """
    Evalúa el modelo.
    """

    model.eval()

    running_loss = 0.0

    correct = 0

    total = 0

    with torch.no_grad():

        for features, labels in dataloader:

            features = features.to(device)

            labels = labels.to(device)

            outputs = model(features)

            loss = criterion(outputs, labels)

            running_loss += loss.item()

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()

            total += labels.size(0)

    epoch_loss = running_loss / len(dataloader)

    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs=30,
    patience=5,
):
    """
    Entrena el modelo utilizando Early Stopping.
    """

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    best_loss = float("inf")

    best_weights = None

    patience_counter = 0

    for epoch in range(epochs):

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        val_loss, val_acc = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch+1:02d}/{epochs}"
            f" | Train Loss: {train_loss:.4f}"
            f" | Val Loss: {val_loss:.4f}"
            f" | Train Acc: {train_acc:.4f}"
            f" | Val Acc: {val_acc:.4f}"
        )

        if val_loss < best_loss:

            best_loss = val_loss

            best_weights = copy.deepcopy(model.state_dict())

            patience_counter = 0

        else:

            patience_counter += 1

        if patience_counter >= patience:

            print("\nEarly Stopping activado.")

            break

    model.load_state_dict(best_weights)

    return model, history