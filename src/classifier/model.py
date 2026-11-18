"""
EI-ARAG Retrieval Decision Classifier Models.
1. RetrievalClassifier: 3-layer MLP (1536 -> 128 -> 64 -> 2)
2. SmallRetrievalClassifier: 2-layer compact MLP (1536 -> 32 -> 2)
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional


class RetrievalClassifier(nn.Module):
    """
    MLP Binary Classifier (3-layer) for Adaptive RAG retrieval decision.
    Architecture: input_dim (1536) -> 128 -> 64 -> 2
    """

    def __init__(
        self,
        input_dim: int = 1536,
        hidden_dim1: int = 128,
        hidden_dim2: int = 64,
        num_classes: int = 2,
        dropout_rate: float = 0.0
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate) if dropout_rate > 0.0 else nn.Identity()

        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate) if dropout_rate > 0.0 else nn.Identity()

        self.fc3 = nn.Linear(hidden_dim2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout1(self.relu1(self.fc1(x)))
        x = self.dropout2(self.relu2(self.fc2(x)))
        logits = self.fc3(x)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=-1)
        return probs

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        probs = self.predict_proba(x)
        return (probs[:, 1] >= threshold).long()

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SmallRetrievalClassifier(nn.Module):
    """
    Compact 2-layer MLP (1536 -> 32 -> 2) designed for small dataset regime (N=70).
    Reduces parameter count by ~76% (49,250 parameters) to mitigate overfitting.
    """

    def __init__(
        self,
        input_dim: int = 1536,
        hidden_dim: int = 32,
        num_classes: int = 2
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=-1)
        return probs

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        probs = self.predict_proba(x)
        return (probs[:, 1] >= threshold).long()

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
