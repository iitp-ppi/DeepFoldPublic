import torch
import torch.nn as nn

from deepfold.modules.alphafold import AlphaFold


class AlphaFoldSWA(nn.Module):
    """Weight averaging wrapper."""

    def __init__(self, alphafold: AlphaFold, enabled: bool, decay_rate: float) -> None:
        super().__init__()

        if enabled:
            self.averaged_model = torch.optim.swa_utils.AveragedModel(
                model=alphafold,
                avg_fn=swa_avg_fn(decay_rate=decay_rate),
            )
        else:
            self.averaged_model = None

    def update(self, model: AlphaFold) -> None:
        if self.averaged_model is not None:
            self.averaged_model.update_parameters(model=model)

    def forward(self, batch):
        if self.averaged_model is None:
            raise RuntimeError("Weight averaging is not enabled")
        return self.averaged_model(batch)


class swa_avg_fn:
    """Averaging function for EMA with configurable decay rate.

    Suppl. '1.11.7 Evaluator setup'.

    """

    def __init__(self, decay_rate: float) -> None:
        self.decay_rate = decay_rate

    def __call__(
        self,
        averaged_model_parameter: torch.Tensor,
        model_parameter: torch.Tensor,
        num_averaged: int,
    ) -> torch.Tensor:
        return averaged_model_parameter + (model_parameter - averaged_model_parameter) * (1.0 - self.decay_rate)
