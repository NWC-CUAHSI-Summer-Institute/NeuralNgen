from typing import Dict, Optional

import torch
import torch.nn as nn


class NgenLSTM(nn.Module):
    """
    Simple LSTM model for space-time hydrology datasets.

    Supports optional static features concatenated to each timestep
    of the dynamic input.

    Parameters
    ----------
    dynamic_input_size : int
        Number of dynamic input features (e.g. time-series variables).
    static_input_size : int, optional
        Number of static input features (e.g. basin attributes). Default = 0.
    hidden_size : int
        Hidden size of the LSTM.
    output_size : int
        Number of output variables (e.g. target variables).
    dropout : float
        Dropout probability applied after the LSTM.
    initial_forget_bias : Optional[float]
        Optional initial forget gate bias.
    """

    def __init__(
        self,
        dynamic_input_size: int,
        static_input_size: int = 0,
        hidden_size: int = 64,
        output_size: int = 1,
        dropout: float = 0.0,
        initial_forget_bias: Optional[float] = None,
    ):
        super().__init__()

        self.static_input_size = static_input_size

        # total input size to LSTM
        lstm_input_size = dynamic_input_size + static_input_size

        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            batch_first=True,
        )

        self.dropout = nn.Dropout(p=dropout)

        # simple linear head
        self.head = nn.Linear(hidden_size, output_size)

        self._initialize(forget_bias=initial_forget_bias)

    def _initialize(self, forget_bias=None):
        # Optional forget gate bias initialization
        if forget_bias is not None:
            # PyTorch stores biases as (b_ih | b_hh), so slice the forget gate bias
            # https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
            hidden_size = self.lstm.hidden_size
            self.lstm.bias_hh_l0.data[hidden_size:2 * hidden_size].fill_(forget_bias)

    def forward(
        self,
        dynamic_inputs: torch.Tensor,
        static_inputs: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        dynamic_inputs : torch.Tensor
            Dynamic input features, shape [batch, seq_len, dynamic_input_size]
        static_inputs : torch.Tensor, optional
            Static input features, shape [batch, static_input_size]

        Returns
        -------
        Dict[str, torch.Tensor]
            - y_hat : predictions, shape [batch, seq_len, output_size]
            - lstm_output : hidden states, shape [batch, seq_len, hidden_size]
            - h_n : last hidden state, shape [batch, hidden_size]
            - c_n : last cell state, shape [batch, hidden_size]
        """

        if static_inputs is not None:
            # expand static inputs across time steps
            B, T, _ = dynamic_inputs.shape
            static_expanded = static_inputs.unsqueeze(1).repeat(1, T, 1)
            lstm_input = torch.cat([dynamic_inputs, static_expanded], dim=-1)
        else:
            lstm_input = dynamic_inputs

        lstm_out, (h_n, c_n) = self.lstm(lstm_input)

        lstm_out = self.dropout(lstm_out)

        y_hat = self.head(lstm_out)

        return {
            "y_hat": y_hat,
            "lstm_output": lstm_out,
            "h_n": h_n.transpose(0, 1),  # [batch, 1, hidden_size]
            "c_n": c_n.transpose(0, 1),
        }