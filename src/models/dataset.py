from sklearn.model_selection import train_test_split


def split_dataset(
    X,
    y,
    test_size=0.20,
    val_size=0.10,
    random_state=42,
):
    """
    Divide el dataset en entrenamiento, validación y prueba.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    val_ratio = val_size / (1 - test_size)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=val_ratio,
        stratify=y_train,
        random_state=random_state,
    )

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    )

import torch

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

def create_tensor_dataset(X, y):
    """
    Convierte la matriz TF-IDF y las etiquetas
    a TensorDataset de PyTorch.
    """

    X_tensor = torch.FloatTensor(X.toarray())
    y_tensor = torch.LongTensor(y.values - 1) # Restamos 1 porque nuestras clases son 1, 2, 3, 4, 5 y CrossEntropyLoss exige que las clases comiencen desde 0.

    return TensorDataset(X_tensor, y_tensor)

def create_dataloader(
    dataset,
    batch_size=64,
    shuffle=False,
):
    """
    Construye un DataLoader.
    """

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )