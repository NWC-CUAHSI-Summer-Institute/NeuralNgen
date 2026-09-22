"""
CatchmentLSTM: the trainable LSTM model wrapped by the BMI interface.

Inputs per catchment/timestep: dynamic forcing variables (time-varying) and
static physiographic attributes (constant per catchment). Static attributes
are broadcast across the sequence length and concatenated with dynamic
inputs at every timestep before entering the LSTM, following the standard
neuralhydrology-style "static + dynamic" LSTM architecture.
"""

import torch
import torch.nn as nn


class CatchmentLSTM(nn.Module):
    def __init__(self, dynamic_size: int, static_size: int,
                 hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.dynamic_size = dynamic_size
        self.static_size = static_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=dynamic_size + static_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, dynamic: torch.Tensor, static: torch.Tensor,
                hidden_state=None):
        """
        dynamic: (G, T, dynamic_size)  -- G catchments, T timesteps
        static:  (G, static_size)
        hidden_state: optional (h, c) tuple of (num_layers, G, hidden_size),
                       used by the BMI wrapper for stepwise/stateful calls.

        Returns: (G, T) predicted output (normalized units), and
                 (h, c) final hidden/cell state.
        """
        g, t, _ = dynamic.shape
        static_expanded = static.unsqueeze(1).expand(g, t, self.static_size)
        x = torch.cat([dynamic, static_expanded], dim=-1)

        out, (h, c) = self.lstm(x, hidden_state)
        pred = self.head(out).squeeze(-1)  # (G, T)
        return pred, (h, c)
