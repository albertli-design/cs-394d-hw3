import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb

from .datasets.classification_dataset import load_data
from .metrics import AccuracyMetric
from .models import load_model, save_model


def train(
    exp_dir: str = "logs",
    model_name: str = "classifier",
    num_epoch: int = 50,
    lr: float = 1e-3,
    batch_size: int = 128,
    seed: int = 2024,
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
        "classification_data/train",
        transform_pipeline=train_transform,
        shuffle=True,
        batch_size=batch_size,
        num_workers=2,
    )
    val_data = load_data(
        "classification_data/val",
        transform_pipeline="default",
        shuffle=False,
        batch_size=batch_size,
        num_workers=2,
    )

    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    global_step = 0
    train_metric = AccuracyMetric()
    val_metric = AccuracyMetric()

    for epoch in range(num_epoch):
        train_metric.reset()
        val_metric.reset()

        model.train()
        for img, label in train_data:
            img, label = img.to(device), label.to(device)

            optimizer.zero_grad()
            logits = model(img)
            loss = loss_func(logits, label)

            logger.add_scalar("train_loss", loss.item(), global_step)

            preds = logits.argmax(dim=1)
            train_metric.add(preds, label)

            loss.backward()
            optimizer.step()

            global_step += 1

        with torch.inference_mode():
            model.eval()
            for img, label in val_data:
                img, label = img.to(device), label.to(device)
                logits = model(img)
                preds = logits.argmax(dim=1)
                val_metric.add(preds, label)

        epoch_train_acc = train_metric.compute()["accuracy"]
        epoch_val_acc = val_metric.compute()["accuracy"]

        logger.add_scalar("train_accuracy", epoch_train_acc, global_step)
        logger.add_scalar("val_accuracy", epoch_val_acc, global_step)

        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
                f"train_acc={epoch_train_acc:.4f} "
                f"val_acc={epoch_val_acc:.4f}"
            )

    save_model(model)
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, default="classifier")
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--train_transform", type=str, default="aug")

    # optional model kwargs (baked into Classifier defaults if omitted)
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--num_classes", type=int, default=6)

    train(**vars(parser.parse_args()))
