import torch


def get_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    if len(optimizer.param_groups) == 1:
        return optimizer.param_groups[0]["lr"]
    elif len(optimizer.param_groups) > 1:
        lr_values_list = [pg["lr"] for pg in optimizer.param_groups]
        lr_values_set = set(lr_values_list)
        if len(lr_values_set) == 1:
            return lr_values_list[0]
        else:
            raise NotImplementedError("Multiple different learning rate values")
    else:
        RuntimeError("Empty `param_groups`")


def set_learning_rate(optimzer: torch.optim.Optimizer, lr_value: float) -> None:
    for param_group in optimzer.param_groups:
        param_group["lr"] = lr_value


class AlphaFoldLRScheduler:
    """AlphaFold learning rate schedule.

    Suppl. '1.11.3 Optimization details'.

    """

    def __init__(
        self,
        init_lr: float,
        final_lr: float,
        warmup_lr_length: int,
        init_lr_length: int,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        self.init_lr = init_lr
        self.final_lr = final_lr
        self.warmup_lr_length = warmup_lr_length
        self.init_lr_length = init_lr_length
        self.optimizer = optimizer

        # Warm-up
        assert warmup_lr_length >= 0
        self.warmup_linspace = torch.linspace(
            start=(init_lr / max(warmup_lr_length, 1)),
            end=init_lr,
            steps=warmup_lr_length,
            device=torch.float64,
        )
        self.prev_lr_value = None

    def __call__(self, iteration: int) -> None:
        if iteration <= self.warmup_lr_length:
            lr_value = self.warmup_linspace[iteration - 1].item()
            lr_value = round(lr_value, 10)
        elif iteration <= self.init_lr_length:
            lr_value = self.init_lr
        else:
            lr_value = self.final_lr
        # Set only if differes from the previous call:
        if lr_value != self.prev_lr_value:
            set_learning_rate(optimzer=self.optimizer, lr_value=lr_value)
            self.prev_lr_value = lr_value
