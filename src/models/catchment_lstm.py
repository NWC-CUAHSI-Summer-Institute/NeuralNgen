"""
CatchmentLSTM

RECONSTRUCTED, not recovered - this file was never pasted into this
conversation (it was lost along with everything else to the /scratch purge).
This is a from-scratch implementation built to satisfy the EXACT call
contract confirmed from your real train_multi_gauge.py:

    model = CatchmentLSTM(dynamic_size=n_dyn, static_size=n_static,
                           hidden_size=cfg.hidden_size, num_layers=cfg.num_layers,
                           dropout_p=cfg.dropout)
    pred = model(x_d, x_s)
        x_d: (N, T_hours, n_dyn)   float32, per-hour dynamic forcings, one row
             per catchment, N = total catchments pooled across all gauges in
             the batch (train_multi_gauge.py does torch.cat over gauges
             before calling the model - the model itself is gauge-agnostic,
             it just sees a flat batch of catchments)
        x_s: (N, n_static)        float32, static attributes, one row per
             catchment (same catchment ordering as x_d)
        pred: (N, T_b)             float32, T_b = T_hours // 24, one
             prediction per calendar day in the window, in NORMALIZED space
             (train_multi_gauge.py de-normalizes it per-gauge afterward using
             that gauge's own q_mean/q_std before area-weighted aggregation)

ARCHITECTURE (the actual reconstruction decisions - flag any of these that
don't match what you remember of the original):

1. Static conditioning: static attributes are encoded through a small MLP
   into the LSTM's initial hidden AND cell state (h0/c0), broadcast across
   all `num_layers`. This is a standard way to inject per-catchment static
   context into an LSTM without concatenating it at every timestep (which
   would also work, but wastes params re-learning the same static effect at
   every hour). If your original model concatenated static attributes to
   the dynamic input at every timestep instead, tell me and I'll switch it -
   that's a real architectural difference, not just a style preference.

2. Hourly -> daily aggregation: the LSTM runs over all T_hours timesteps
   (hourly resolution, matching your hourly AORC forcings), producing one
   hidden state per hour. To get T_b daily predictions, hidden states are
   MEAN-POOLED within each 24-hour block, then passed through the output
   head. This means each day's prediction is informed by that whole day's
   hourly dynamics, not just its last hour. An alternative (take only the
   last hour of each day) is also defensible - flag if you recall which one
   the original used.

3. Output head: Dropout -> Linear(hidden_size, 1), applied per day-block,
   squeezed to (N, T_b). Single-target regression per catchment per day,
   consistent with the scoring-rule losses operating on (n_gauges, T_b)
   after aggregation.

4. No activation is applied to the raw output (it stays in normalized
   z-score space) - train_multi_gauge.py explicitly de-normalizes with
   `pred_chunk * q_std + q_mean` and then clamps to >= 0 in PHYSICAL space,
   so clamping here (in normalized space) would be wrong.
"""

import torch
import torch.nn as nn


class CatchmentLSTM(nn.Module):
    def __init__(self, dynamic_size: int, static_size: int,
                 hidden_size: int = 64, num_layers: int = 2, dropout_p: float = 0.2):
        super().__init__()
        self.dynamic_size = dynamic_size
        self.static_size = static_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Static attributes -> initial (h0, c0) for every LSTM layer.
        # Two independent small MLPs (not shared weights) since h0 and c0
        # play different roles in the LSTM recurrence.
        self.static_to_h0 = nn.Sequential(
            nn.Linear(static_size, hidden_size),
            nn.Tanh(),
        )
        self.static_to_c0 = nn.Sequential(
            nn.Linear(static_size, hidden_size),
            nn.Tanh(),
        )

        self.lstm = nn.LSTM(
            input_size=dynamic_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_p if num_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(hidden_size, 1),
        )

        # --- stateful single-timestep inference (BMI/ngen) ---
        # Buffers below are only ever touched by forward_step(); forward()
        # (used for training/offline evaluation) never reads or writes them.
        self._day_hidden_buffer: list[torch.Tensor] = []
        self._hour_in_day: int = 0
        self._last_pred: torch.Tensor | None = None

    def forward(self, x_d: torch.Tensor, x_s: torch.Tensor) -> torch.Tensor:
        n, t_hours, _ = x_d.shape
        if t_hours % 24 != 0:
            raise ValueError(
                f"CatchmentLSTM expects an hourly window that is a whole "
                f"number of days (multiple of 24 hours); got {t_hours}"
            )
        t_b = t_hours // 24

        h0 = self.static_to_h0(x_s).unsqueeze(0).repeat(self.num_layers, 1, 1).contiguous()
        c0 = self.static_to_c0(x_s).unsqueeze(0).repeat(self.num_layers, 1, 1).contiguous()

        # out: (N, T_hours, hidden_size)
        out, _ = self.lstm(x_d, (h0, c0))

        # Mean-pool hidden states within each 24-hour block -> (N, T_b, hidden_size)
        out_daily = out.view(n, t_b, 24, self.hidden_size).mean(dim=2)

        pred = self.head(out_daily).squeeze(-1)  # (N, T_b)
        return pred

    def forward_step(
        self,
        x_d_t: torch.Tensor,
        x_s: torch.Tensor,
        h_t: "torch.Tensor | None" = None,
        c_t: "torch.Tensor | None" = None,
    ) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
        """
        Stateful single-HOUR step, for callers (ngen/BMI) that feed one
        hour of forcing at a time and expect one prediction back per call.

        This is NOT a different model - it reproduces forward() exactly,
        just spread across 24 calls instead of one. The LSTM cell genuinely
        advances every hour (h_t/c_t are real recurrent state, carried by
        the caller between calls, same as forward() carries them internally
        across a window). But the output head only ever sees a full day's
        mean-pooled hidden state, because that's the only input it was
        trained on - so it only actually fires once every 24 calls, on the
        mean of that day's 24 hourly hidden states. The other 23 calls
        re-emit the previous completed day's prediction unchanged: there is
        no faithful hourly-resolution value to give instead, since the head
        has no learned response to a single hour's hidden state.

        Parameters
        ----------
        x_d_t : (N, dynamic_size) - one hour of dynamic forcing.
        x_s   : (N, static_size)  - static attributes. Only actually used
            at the START of a new day-buffer (i.e. right after a reset),
            to seed (h0, c0) exactly like forward() does. Pass it every
            call regardless - it's cheap and keeps the call contract
            uniform; it's ignored on the 23 calls where it isn't needed.
        h_t, c_t : (num_layers, N, hidden_size) previous LSTM state, or
            None to start a fresh run (initializes state from x_s AND
            resets the day buffer - use this exactly once, on the very
            first call of a simulation for a given catchment instance).

        Returns
        -------
        pred  : (N, 1) NORMALIZED prediction (z-score space, same as
            forward()'s output before any de-normalization). Changes once
            every 24 calls; held constant in between.
        h_new : (num_layers, N, hidden_size) - pass back in next call.
        c_new : (num_layers, N, hidden_size) - pass back in next call.
        """
        if h_t is None or c_t is None:
            h_t = self.static_to_h0(x_s).unsqueeze(0).repeat(self.num_layers, 1, 1).contiguous()
            c_t = self.static_to_c0(x_s).unsqueeze(0).repeat(self.num_layers, 1, 1).contiguous()
            self._day_hidden_buffer = []
            self._hour_in_day = 0
            self._last_pred = None

        x_d_step = x_d_t.unsqueeze(1)  # (N, 1, dynamic_size)
        out, (h_new, c_new) = self.lstm(x_d_step, (h_t, c_t))
        out = out.squeeze(1)  # (N, hidden_size)

        self._day_hidden_buffer.append(out)
        self._hour_in_day += 1

        if self._hour_in_day == 24:
            day_mean = torch.stack(self._day_hidden_buffer, dim=1).mean(dim=1)  # (N, hidden_size)
            pred = self.head(day_mean)  # (N, 1)
            self._last_pred = pred
            self._day_hidden_buffer = []
            self._hour_in_day = 0
        else:
            if self._last_pred is None:
                # First day of the run, still mid-day: nothing has been
                # emitted yet. Zero is the only non-fabricated choice -
                # anything else would imply the head evaluated a partial
                # day, which it never does.
                self._last_pred = x_d_t.new_zeros(x_d_t.shape[0], 1)
            pred = self._last_pred

        return pred, h_new, c_new