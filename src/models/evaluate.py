import torch

import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


def predict(model, dataloader, device):
    """
    Obtiene las predicciones del modelo.
    """

    model.eval()

    predictions = []
    targets = []

    with torch.no_grad():

        for features, labels in dataloader:

            features = features.to(device)

            outputs = model(features)

            preds = outputs.argmax(dim=1)

            predictions.extend(preds.cpu().numpy())

            targets.extend(labels.numpy())

    return targets, predictions


def evaluate_model(
    model,
    dataloader,
    device,
):
    """
    Evalúa el modelo sobre un conjunto de datos y devuelve las principales
    métricas de clasificación junto con las etiquetas reales y predichas.    
    """

    y_true, y_pred = predict(
        model,
        dataloader,
        device,
    )

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    report = classification_report(
        y_true,
        y_pred,
        digits=4,
    )

    return accuracy, report, y_true, y_pred


def plot_confusion_matrix(
    y_true,
    y_pred,
):
    """
    Grafica la matriz de confusión.
    """

    cm = confusion_matrix(
        y_true,
        y_pred,
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
    )

    fig, ax = plt.subplots(figsize=(7,6))

    disp.plot(
        cmap="Blues",
        ax=ax,
        colorbar=False,
    )

    plt.title("Matriz de confusión")

    plt.show()


def plot_history(history):
    """
    Grafica la evolución del entrenamiento.
    """

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12,4),
    )

    axes[0].plot(
        history["train_loss"],
        label="Train",
    )

    axes[0].plot(
        history["val_loss"],
        label="Validation",
    )

    axes[0].set_title("Loss")

    axes[0].legend()

    axes[1].plot(
        history["train_acc"],
        label="Train",
    )

    axes[1].plot(
        history["val_acc"],
        label="Validation",
    )

    axes[1].set_title("Accuracy")

    axes[1].legend()

    plt.tight_layout()

    plt.show()