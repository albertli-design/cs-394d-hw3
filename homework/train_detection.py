import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.tensorboard as tb

from .datasets.road_dataset import load_data
from .metrics import DetectionMetric
from .models import load_model, save_model


def train(
    exp_dir: str = "logs",
    model_name: str = "detector",
    num_epoch: int = 40,
    lr: float = 1e-3,
    batch_size: int = 32,
    seed: int = 2024,
    depth_weight: float = 0.5,
    train_transform: str = "aug",
    **kwargs,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)

    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    model = load_model(model_name, **kwargs)
    model = model.to(device)
    model.train()

    train_data = load_data(
        "drive_data/train",
        transform_pipeline=train_transform,
        shuffle=True,
        batch_size=batch_size,
        num_workers=2,
    )
    val_data = load_data(
        "drive_data/val",
        transform_pipeline="default",
        shuffle=False,
        batch_size=batch_size,
        num_workers=2,
    )

    # background dominates; upweight thin left/right lane classes for IoU
    class_weights = torch.tensor([1.0, 12.0, 12.0], device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epoch)

    global_step = 0
    train_metric = DetectionMetric()
    val_metric = DetectionMetric()
    best_val_iou = -1.0

    for epoch in range(num_epoch):
        train_metric.reset()
        val_metric.reset()

        model.train()
        for batch in train_data:
            image = batch["image"].to(device)
            depth = batch["depth"].to(device)
            track = batch["track"].to(device)

            optimizer.zero_grad()
            logits, depth_pred = model(image)

            seg_loss = F.cross_entropy(logits, track, weight=class_weights)
            depth_loss = F.l1_loss(depth_pred, depth)
            loss = seg_loss + depth_weight * depth_loss

            logger.add_scalar("train_loss", loss.item(), global_step)
            logger.add_scalar("train_seg_loss", seg_loss.item(), global_step)
            logger.add_scalar("train_depth_loss", depth_loss.item(), global_step)

            preds = logits.argmax(dim=1)
            train_metric.add(preds, track, depth_pred.detach(), depth)

            loss.backward()
            optimizer.step()

            global_step += 1

        with torch.inference_mode():
            model.eval()
            for batch in val_data:
                image = batch["image"].to(device)
                depth = batch["depth"].to(device)
                track = batch["track"].to(device)

                logits, depth_pred = model(image)
                preds = logits.argmax(dim=1)
                val_metric.add(preds, track, depth_pred, depth)

        train_stats = train_metric.compute()
        val_stats = val_metric.compute()
        scheduler.step()

        logger.add_scalar("train_iou", train_stats["iou"], global_step)
        logger.add_scalar("train_accuracy", train_stats["accuracy"], global_step)
        logger.add_scalar("val_iou", val_stats["iou"], global_step)
        logger.add_scalar("val_accuracy", val_stats["accuracy"], global_step)
        logger.add_scalar("val_abs_depth_error", val_stats["abs_depth_error"], global_step)
        logger.add_scalar("val_tp_depth_error", val_stats["tp_depth_error"], global_step)
        logger.add_scalar("lr", scheduler.get_last_lr()[0], global_step)

        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 5 == 0:
            print(
                f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
                f"train_iou={train_stats['iou']:.4f} "
                f"val_iou={val_stats['iou']:.4f} "
                f"val_acc={val_stats['accuracy']:.4f} "
                f"depth_err={val_stats['abs_depth_error']:.4f} "
                f"tp_depth_err={val_stats['tp_depth_error']:.4f}"
            )

        # keep best-by-IoU weights for grading (depth already strong)
        if val_stats["iou"] > best_val_iou:
            best_val_iou = val_stats["iou"]
            save_model(model)
            torch.save(model.state_dict(), log_dir / f"{model_name}.th")
            print(f"  new best val_iou={best_val_iou:.4f} -> saved")

    print(f"Best val_iou={best_val_iou:.4f}")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, default="detector")
    parser.add_argument("--num_epoch", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--depth_weight", type=float, default=0.5)
    parser.add_argument("--train_transform", type=str, default="aug")

    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--num_classes", type=int, default=3)

    train(**vars(parser.parse_args()))
