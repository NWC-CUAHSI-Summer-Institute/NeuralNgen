import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class ngenLoss(nn.Module):
    """
    Combined loss function:
    - Residual loss: MSE, RMSE, or NSE
    - Variogram loss (spatial)
    - FDC divergence loss

    Parameters
    ----------
    distance_matrix : torch.Tensor or None
        Pairwise distance matrix (sites x sites). Can be None to skip weighting.
    variogram_weight : float
        Weight for the variogram loss.
    fdc_weight : float
        Weight for the FDC loss.
    v_order : int
        Power for the variogram (default=2 for semi-variance).
    residualloss : str
        Type of residual loss. One of {"mse", "rmse", "nse"}.
    """

    def __init__(
        self,
        cfg,
        distance_matrix=None,
        v_order=2,
    ):
        super().__init__()

        self.distance_matrix = distance_matrix
        self.variogram_weight = cfg.variogram_weight
        self.fdc_weight = cfg.fdc_weight
        self.residual_loss = cfg.residual_loss
        self.residual_weight = cfg.residual_weight
        self.v_order = v_order

        assert self.residual_loss.lower() in ["mse", "rmse", "nse"], \
            f"Invalid residualloss: {self.residual_loss}"
        self.residual_loss = self.residual_loss.lower()

    def forward(self, prediction, data):
        """
        Compute total loss.

        Parameters
        ----------
        prediction : Dict[str, torch.Tensor]
            Must contain key "y_hat" [B, T, D]
        data : Dict[str, torch.Tensor]
            Must contain key "y" [B, T, D]

        Returns
        -------
        torch.Tensor
            Scalar loss value.
        dict
            Dictionary of loss components.
        """

        y_hat = prediction["y_hat"]
        y = data["y"]

        # Mask NaNs
        mask = ~torch.isnan(y)
        y_hat_masked = torch.where(mask, y_hat, torch.zeros_like(y_hat))
        y_masked = torch.where(mask, y, torch.zeros_like(y))

        loss=0
        losses = {}

        if self.residual_weight > 0:
            residual = self._compute_residual_loss(y_hat_masked, y_masked, mask)
            loss += self.residual_weight * residual
            losses[f"{self.residual_loss}_loss"] = residual
        if self.variogram_weight > 0:
            vario_loss = self._variogram_loss(y_hat, y, mask)
            loss += self.variogram_weight * vario_loss
            losses["variogram_loss"] = vario_loss
        if self.fdc_weight > 0:
            fdc_loss = self._fdc_loss(y_hat, y, mask)
            loss += self.fdc_weight * fdc_loss
            losses["fdc_loss"] = fdc_loss

        losses["total_loss"] = loss

        return loss, losses

    def _compute_residual_loss(self, y_hat, y, mask):
        """
        Compute residual loss according to selected type.

        Parameters
        ----------
        y_hat : torch.Tensor
            [B, T, D] predicted
        y : torch.Tensor
            [B, T, D] observed
        mask : torch.Tensor
            Boolean mask, same shape as y

        Returns
        -------
        torch.Tensor
            Scalar residual loss
        """

        # Flatten only valid entries
        y_hat_flat = y_hat[mask]
        y_flat = y[mask]

        if y_flat.numel() == 0:
            # All missing data
            return torch.tensor(0.0, device=y.device)

        if self.residual_loss == "mse":
            loss_value = F.mse_loss(y_hat_flat, y_flat)

        elif self.residual_loss == "rmse":
            loss_value = torch.sqrt(F.mse_loss(y_hat_flat, y_flat))

        elif self.residual_loss == "nse":
            # Nash–Sutcliffe efficiency with small eps to avoid division by zero
            # Flatten only valid entries
            residuals = y_flat - y_hat_flat
            sse = torch.sum(residuals ** 2)
            variance = torch.var(y_flat, unbiased=False)

            # add tiny epsilon so we never divide by zero
            eps = 1e-6
            denom = variance * y_flat.numel() + eps

            nse = 1 - (sse / denom)
            # Convert to a loss: lower NSE → higher loss, clamp at 0
            loss_value = torch.clamp(1.0 - nse, min=0.0)

        else:
            raise ValueError(f"Unknown residual loss type: {self.residualloss}")

        return loss_value

    def _variogram_loss(self, y_hat, y, mask):
        """
        Compute spatial variogram loss.

        Expects y_hat and y of shape [B, T, D=1]
        """
        method = "quantile"
        y_hat_flat = self.var_hyd_char(y_hat, method=method, 
                                        random_quantile=True).squeeze(-1)
        y_flat = self.var_hyd_char(y, method=method, 
                                      random_quantile=True).squeeze(-1)

        # Compute pairwise |yi - yj|^v
        diff_obs = torch.abs(y_flat[:, None] - y_flat[None, :]) ** self.v_order
        diff_pred = torch.abs(y_hat_flat[:, None] - y_hat_flat[None, :]) ** self.v_order

        loss_matrix = (diff_obs - diff_pred) ** 2

        if self.distance_matrix is not None:
            weight = self.distance_matrix.to(y.device)
            loss_matrix = loss_matrix * weight

        return loss_matrix.mean()

    def _fdc_loss(self, y_hat, y, mask):
        """
        Compute FDC divergence loss.
        """

        # Collapse sites and time to a flat vector for FDC
        y_hat_flat = y_hat[mask].flatten()
        y_flat = y[mask].flatten()

        if y_flat.numel() == 0:
            return torch.tensor(0.0, device=y.device)

        y_hat_sorted, _ = torch.sort(y_hat_flat, descending=True)
        y_sorted, _ = torch.sort(y_flat, descending=True)

        min_len = min(len(y_sorted), len(y_hat_sorted))
        y_hat_sorted = y_hat_sorted[:min_len]
        y_sorted = y_sorted[:min_len]

        # Cross variability
        cross_diff = torch.abs(y_sorted[:, None] - y_hat_sorted[None, :])
        cross_var = cross_diff.mean()

        # Within-variability
        within_y = torch.abs(y_sorted[:, None] - y_sorted[None, :]).mean()
        within_hat = torch.abs(y_hat_sorted[:, None] - y_hat_sorted[None, :]).mean()

        fdc_divergence = cross_var - 0.5 * (within_y + within_hat)
        fdc_divergence = torch.clamp(fdc_divergence, min=0.0)

        return fdc_divergence

    @staticmethod
    def var_hyd_char(
        ts: torch.Tensor, 
        method: str = "mean", 
        quantile: float = 0.25,
        random_quantile: bool = False,
        quantile_bounds: tuple = (0.05, 0.95)
    ):
        """
        Compute a single hydrograph characteristic from time series for each basin.

        Parameters
        ----------
        ts : torch.Tensor
            Time series, shape [B, T, D]
        method : str
            Aggregation method. Options:
                - "mean"
                - "median"
                - "std"
                - "quantile"
        quantile : float
            If method == "quantile", the quantile level (0..1).
        random_quantile : bool
            Whether to randomly choose a quantile between quantile_bounds.
        quantile_bounds : tuple(float, float)
            Lower and upper bounds for random quantile.

        Returns
        -------
        torch.Tensor
            Tensor of shape [B, D]
        """

        if method == "mean":
            return ts.mean(dim=1)

        elif method == "median":
            return ts.median(dim=1).values

        elif method == "std":
            return ts.std(dim=1)

        elif method == "quantile":
            if random_quantile:
                quantile = random.uniform(*quantile_bounds)
            sorted_ts, _ = ts.sort(dim=1, descending=True)
            index = int(quantile * ts.shape[1])
            index = max(0, min(index, ts.shape[1] - 1))
            return sorted_ts[:, index, :]
        
        else:
            raise ValueError(f"Unknown method for hydrograph characteristic: {method}")