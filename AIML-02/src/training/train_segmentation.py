"""Train a 3D U-Net on the Medical Segmentation Decathlon (MONAI).

Requires monai + torch (Colab/GPU). Downloads a Decathlon task (default Task09
Spleen - small, single-organ) on first run, trains the U-Net with Dice loss, and
reports the Dice metric. This is the brief's segmentation deliverable; it is
provided as runnable code, trained on Colab rather than in the build sandbox.

Run:
    python -m src.training.train_segmentation --task Task09_Spleen --epochs 30
"""
from __future__ import annotations

import argparse

from ..common import config as C


def train(task: str = "Task09_Spleen", epochs: int = 30, root: str = "data/decathlon"):
    import torch
    from monai.apps import DecathlonDataset
    from monai.data import DataLoader, decollate_batch
    from monai.inferers import sliding_window_inference
    from monai.losses import DiceLoss
    from monai.metrics import DiceMetric
    from monai.transforms import (Activations, AsDiscrete, Compose,
                                  EnsureChannelFirstd, EnsureTyped, LoadImaged,
                                  Orientationd, RandCropByPosNegLabeld,
                                  ScaleIntensityRanged, Spacingd)

    from ..models.segmentation import build_unet

    train_tf = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0),
                 mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=-57, a_max=164,
                             b_min=0.0, b_max=1.0, clip=True),
        RandCropByPosNegLabeld(keys=["image", "label"], label_key="label",
                               spatial_size=C.SEG_ROI_SIZE, pos=1, neg=1, num_samples=4),
        EnsureTyped(keys=["image", "label"]),
    ])
    val_tf = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0),
                 mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=-57, a_max=164,
                             b_min=0.0, b_max=1.0, clip=True),
        EnsureTyped(keys=["image", "label"]),
    ])

    train_ds = DecathlonDataset(root_dir=root, task=task, section="training",
                                transform=train_tf, download=True, cache_rate=0.0)
    val_ds = DecathlonDataset(root_dir=root, task=task, section="validation",
                              transform=val_tf, download=False, cache_rate=0.0)
    train_dl = DataLoader(train_ds, batch_size=2, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_unet().to(device)
    loss_fn = DiceLoss(to_onehot_y=True, softmax=True)
    opt = torch.optim.Adam(model.parameters(), 1e-4)
    dice = DiceMetric(include_background=False, reduction="mean")
    post_pred = Compose([Activations(softmax=True), AsDiscrete(argmax=True, to_onehot=2)])
    post_label = Compose([AsDiscrete(to_onehot=2)])

    for epoch in range(epochs):
        model.train()
        for batch in train_dl:
            x, y = batch["image"].to(device), batch["label"].to(device)
            opt.zero_grad()
            loss_fn(model(x), y).backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        for batch in val_dl:
            x, y = batch["image"].to(device), batch["label"].to(device)
            out = sliding_window_inference(x, C.SEG_ROI_SIZE, 4, model)
            out = [post_pred(o) for o in decollate_batch(out)]
            lab = [post_label(o) for o in decollate_batch(y)]
            dice(y_pred=out, y=lab)
    score = float(dice.aggregate().item())
    dice.reset()
    torch.save({"model": model.state_dict()}, C.SEG_MODEL_PATH)
    print({"task": task, "val_dice": score})
    return {"task": task, "val_dice": score}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="Task09_Spleen")
    p.add_argument("--epochs", type=int, default=30)
    a = p.parse_args()
    train(a.task, a.epochs)
