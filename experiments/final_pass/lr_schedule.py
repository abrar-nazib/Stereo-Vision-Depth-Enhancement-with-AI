"""Learning-rate policies for the full-SceneFlow comparison runs."""

import torch


def make_scheduler(name: str, optimizer, total_steps: int, max_lr: float):
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3, threshold=0.005,
            threshold_mode="rel", cooldown=1, min_lr=1e-6)
    if name == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=max_lr, total_steps=total_steps, pct_start=0.01,
            anneal_strategy="cos", div_factor=25.0, final_div_factor=4.0,
            cycle_momentum=False)
    raise ValueError(f"unknown LR schedule: {name}")


def after_training_step(name: str, scheduler, optimizer_updated: bool = True) -> None:
    if name == "onecycle" and optimizer_updated:
        scheduler.step()


def after_validation(name: str, scheduler, val_epe: float) -> None:
    if name == "plateau":
        scheduler.step(val_epe)


def description(name: str, max_lr: float, total_steps: int) -> str:
    if name == "plateau":
        return ("ReduceLROnPlateau on validation EPE: factor=0.5, patience=3 "
                "evaluations, relative threshold=0.005, cooldown=1, min_lr=1e-6")
    if name == "onecycle":
        return (f"OneCycleLR per optimizer step: max_lr={max_lr:g}, "
                f"total_steps={total_steps}, pct_start=0.01, cosine, "
                "div_factor=25, final_div_factor=4, cycle_momentum=False")
    raise ValueError(f"unknown LR schedule: {name}")
