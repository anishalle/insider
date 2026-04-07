from __future__ import annotations

import torch
from torch import nn


class RNNBinaryClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        effective_dropout = dropout if num_layers > 1 else 0.0
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity="tanh",
            dropout=effective_dropout,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.dim() != 3:
            raise ValueError("Expected inputs with shape (batch, sequence, features).")
        _, hidden = self.rnn(inputs)
        final_hidden = hidden[-1]
        logits = self.head(final_hidden)
        return logits.squeeze(-1)

