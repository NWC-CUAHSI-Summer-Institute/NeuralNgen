# Training the model

## Persistent Training (Majid's Approach)

Set `persistent_state: true` in your config to enable persistent LSTM training.

### What it does

When **persistent_state** is enabled, the trainer:

1. **Chronological batching** — within each basin, batches are always in time order (A1 → A2 → A3), while basins are randomly interleaved across each other
2. **Hidden state carryover** — the LSTM's hidden state (h, c) is cached per basin and carried from one batch to the next within each epoch
3. **Epoch reset** — hidden states are cleared at the start of each epoch (no cross-epoch persistence)
4. **Time-batch flattening** — batches are reshaped from [B, L, ...] to [1, B*L, ...] so the model processes them as one continuous time segment

### Why it matters

Standard training randomizes batch order and resets hidden state each batch. This prevents the LSTM from learning long-term temporal dependencies. Persistent training preserves temporal continuity, which is critical for hydrology (e.g., soil moisture memory, snowpack accumulation).

### Reference

Based on [Majid's neuralhydrology-persistent](https://github.com/majid828/neuralhydrology-persistent/blob/master/neuralhydrology/training/basetrainer.py).
