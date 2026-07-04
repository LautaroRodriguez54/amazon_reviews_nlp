"""
Modelo de clasificación de texto mediante una red neuronal.
"""

import torch.nn as nn


class TextClassifier(nn.Module):
    """
    Red neuronal para clasificación de texto basada en TF-IDF.
    """

    def __init__(self):
        super().__init__()

import torch.nn as nn


class TextClassifier(nn.Module):
    """
    Red neuronal para clasificación de texto basada en TF-IDF.
    """

    def __init__(
        self,
        input_dim: int = 5000,
        hidden_dim: int = 256,
        num_classes: int = 5,
    ):
        super().__init__()

        self.fc1 = nn.Linear(
            input_dim,
            hidden_dim
        )

import torch.nn as nn


class TextClassifier(nn.Module):

    def __init__(
        self,
        input_dim: int = 5000,
        hidden_dim: int = 256,
        num_classes: int = 5,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)

        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):

        x = self.fc1(x)

        x = self.bn1(x)

        x = self.relu(x)

        x = self.dropout(x)

        x = self.fc2(x)

        return x

