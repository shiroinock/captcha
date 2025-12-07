import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, random_split

from dataset import (
    CHARS,
    CaptchaDataset,
    get_train_transforms,
    get_val_transforms,
)
from model import FastCaptchaSolver


def create_run_dir(base_dir: str = "runs") -> Path:
    """Create a timestamped run directory for saving training artifacts."""
    base_path = Path(base_dir)
    base_path.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_path / timestamp
    run_dir.mkdir(exist_ok=True)
    return run_dir


def save_training_config(run_dir: Path, args: argparse.Namespace, extra_info: dict = None):
    """Save training configuration to JSON."""
    config = vars(args).copy()
    config["timestamp"] = datetime.now().isoformat()
    if extra_info:
        config.update(extra_info)
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)


def plot_training_curves(run_dir: Path, history: list[dict]):
    """Generate and save training curves as PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plot generation")
        return

    epochs = [h["epoch"] for h in history]

    _, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Loss
    axes[0].plot(epochs, [h["train_loss"] for h in history], label="Train")
    axes[0].plot(epochs, [h["val_loss"] for h in history], label="Val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True)

    # Char Accuracy
    axes[1].plot(epochs, [h["train_char_acc"] for h in history], label="Train")
    axes[1].plot(epochs, [h["val_char_acc"] for h in history], label="Val")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Character Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    # Seq Accuracy
    axes[2].plot(epochs, [h["train_seq_acc"] for h in history], label="Train")
    axes[2].plot(epochs, [h["val_seq_acc"] for h in history], label="Val")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Accuracy")
    axes[2].set_title("Sequence Accuracy")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(run_dir / "training_curves.png", dpi=150)
    plt.close()


def save_results_summary(run_dir: Path, best_metrics: dict, final_epoch: int):
    """Save final results summary."""
    summary = {
        "best_val_seq_acc": best_metrics["val_seq_acc"],
        "best_val_char_acc": best_metrics["val_char_acc"],
        "best_epoch": best_metrics["epoch"],
        "final_epoch": final_epoch,
    }
    with open(run_dir / "results.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Also print a summary
    print("\n" + "=" * 50)
    print("Training Summary")
    print("=" * 50)
    print(f"Run directory: {run_dir}")
    print(f"Best Val Seq Acc: {best_metrics['val_seq_acc']:.4f} (epoch {best_metrics['epoch']})")
    print(f"Best Val Char Acc: {best_metrics['val_char_acc']:.4f}")
    print(f"Total epochs: {final_epoch}")
    print("=" * 50)


class FocalLoss(nn.Module):
    """Focal Loss for handling hard examples (e.g., confusable characters O/0, l/1/I)."""

    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(
        self, inputs: torch.Tensor, targets: torch.Tensor, label_smoothing: float = 0.0
    ) -> torch.Tensor:
        ce_loss = F.cross_entropy(
            inputs, targets, reduction="none", label_smoothing=label_smoothing
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def compute_loss(
    outputs: torch.Tensor,
    labels: torch.Tensor,
    num_chars: int,
    label_smoothing: float = 0.0,
    focal_loss: FocalLoss | None = None,
) -> torch.Tensor:
    if focal_loss is not None:
        loss = sum(
            focal_loss(outputs[:, i, :], labels[:, i], label_smoothing)
            for i in range(num_chars)
        )
    else:
        loss = sum(
            F.cross_entropy(outputs[:, i, :], labels[:, i], label_smoothing=label_smoothing)
            for i in range(num_chars)
        )
    return loss


def compute_accuracy(
    outputs: torch.Tensor, labels: torch.Tensor
) -> tuple[float, float]:
    preds = outputs.argmax(dim=-1)
    char_correct = (preds == labels).float().mean().item()
    seq_correct = (preds == labels).all(dim=1).float().mean().item()
    return char_correct, seq_correct


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_chars: int,
    label_smoothing: float = 0.0,
    focal_loss: FocalLoss | None = None,
) -> tuple[float, float, float]:
    model.train()
    total_loss = 0.0
    total_char_acc = 0.0
    total_seq_acc = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = compute_loss(outputs, labels, num_chars, label_smoothing, focal_loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        char_acc, seq_acc = compute_accuracy(outputs, labels)
        total_loss += loss.item()
        total_char_acc += char_acc
        total_seq_acc += seq_acc

    n = len(loader)
    return total_loss / n, total_char_acc / n, total_seq_acc / n


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_chars: int,
) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total_char_acc = 0.0
    total_seq_acc = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = compute_loss(outputs, labels, num_chars)

        char_acc, seq_acc = compute_accuracy(outputs, labels)
        total_loss += loss.item()
        total_char_acc += char_acc
        total_seq_acc += seq_acc

    n = len(loader)
    return total_loss / n, total_char_acc / n, total_seq_acc / n


def main():
    parser = argparse.ArgumentParser(description="Train CAPTCHA OCR model")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="images",
        help="Path to images directory",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--warmup-epochs", type=int, default=5, help="LR warmup epochs")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument(
        "--save-dir", type=str, default="models", help="Directory to save models"
    )
    parser.add_argument(
        "--label-smoothing", type=float, default=0.05, help="Label smoothing factor"
    )
    parser.add_argument(
        "--focal-loss", action="store_true", help="Use Focal Loss for hard examples"
    )
    parser.add_argument(
        "--focal-gamma", type=float, default=2.0, help="Focal Loss gamma parameter"
    )
    parser.add_argument(
        "--run-dir", type=str, default="runs", help="Base directory for training runs"
    )
    parser.add_argument(
        "--name", type=str, default=None, help="Optional run name suffix"
    )
    args = parser.parse_args()

    # Create run directory for this training session
    run_dir = create_run_dir(args.run_dir)
    if args.name:
        # Rename with suffix
        new_run_dir = run_dir.parent / f"{run_dir.name}_{args.name}"
        run_dir.rename(new_run_dir)
        run_dir = new_run_dir
    print(f"Run directory: {run_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    num_chars = 5
    num_classes = len(CHARS)

    full_dataset = CaptchaDataset(
        args.data_dir, transform=get_train_transforms(), num_chars=num_chars
    )
    print(f"Total images: {len(full_dataset)}")

    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    val_dataset.dataset = CaptchaDataset(
        args.data_dir, transform=get_val_transforms(), num_chars=num_chars
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    model = FastCaptchaSolver(num_chars=num_chars, num_classes=num_classes).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # LR Warmup + Cosine Annealing
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=args.warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs)
    scheduler = SequentialLR(
        optimizer, [warmup_scheduler, cosine_scheduler], milestones=[args.warmup_epochs]
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)

    # Initialize Focal Loss if enabled
    focal_loss = FocalLoss(gamma=args.focal_gamma) if args.focal_loss else None
    if focal_loss:
        print(f"Using Focal Loss (gamma={args.focal_gamma})")

    # Save training configuration
    save_training_config(run_dir, args, {
        "num_classes": num_classes,
        "num_chars": num_chars,
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
    })

    # Initialize CSV logger
    csv_path = run_dir / "history.csv"
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=[
        "epoch", "train_loss", "train_char_acc", "train_seq_acc",
        "val_loss", "val_char_acc", "val_seq_acc", "lr"
    ])
    csv_writer.writeheader()

    history = []
    best_seq_acc = 0.0
    best_metrics = {}
    patience_counter = 0

    for epoch in range(args.epochs):
        train_loss, train_char_acc, train_seq_acc = train_one_epoch(
            model, train_loader, optimizer, device, num_chars, args.label_smoothing, focal_loss
        )
        val_loss, val_char_acc, val_seq_acc = validate(
            model, val_loader, device, num_chars
        )
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        # Log to CSV
        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_char_acc": train_char_acc,
            "train_seq_acc": train_seq_acc,
            "val_loss": val_loss,
            "val_char_acc": val_char_acc,
            "val_seq_acc": val_seq_acc,
            "lr": current_lr,
        }
        csv_writer.writerow(row)
        csv_file.flush()
        history.append(row)

        timestamp = datetime.now().strftime("%H:%M:%S")
        print(
            f"[{timestamp}] Epoch {epoch + 1:02d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f}, Char: {train_char_acc:.4f}, Seq: {train_seq_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Char: {val_char_acc:.4f}, Seq: {val_seq_acc:.4f}"
        )

        if val_seq_acc > best_seq_acc:
            best_seq_acc = val_seq_acc
            best_metrics = {
                "epoch": epoch + 1,
                "val_seq_acc": val_seq_acc,
                "val_char_acc": val_char_acc,
            }
            patience_counter = 0
            # Save to both run_dir and save_dir
            torch.save(model.state_dict(), run_dir / "best_model.pt")
            torch.save(model.state_dict(), save_dir / "best_model.pt")
            print(f"  -> Saved best model (seq_acc: {best_seq_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break

    csv_file.close()

    # Save final model
    torch.save(model.state_dict(), run_dir / "final_model.pt")
    torch.save(model.state_dict(), save_dir / "final_model.pt")

    # Generate training curves
    plot_training_curves(run_dir, history)

    # Save results summary
    save_results_summary(run_dir, best_metrics, epoch + 1)


if __name__ == "__main__":
    main()
