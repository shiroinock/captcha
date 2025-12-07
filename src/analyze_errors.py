"""Analyze model errors by visualizing misclassified CAPTCHA images."""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from dataset import CHARS, CaptchaDataset, get_val_transforms
from model import FastCaptchaSolver


def decode_prediction(indices: torch.Tensor) -> str:
    """Convert model output indices to string."""
    return "".join(CHARS[i] for i in indices)


def analyze_errors(
    model_path: str,
    data_dir: str,
    output_dir: str,
    num_samples: int = 50,
    val_split: float = 0.1,
):
    """Find and visualize misclassified images."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required: uv pip install matplotlib")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    num_chars = 5
    num_classes = len(CHARS)
    model = FastCaptchaSolver(num_chars=num_chars, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print(f"Loaded model from {model_path}")

    # Load dataset (same split as training)
    full_dataset = CaptchaDataset(data_dir, transform=get_val_transforms(), num_chars=num_chars)
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    _, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"Validation set size: {len(val_dataset)}")

    # Collect errors
    errors = []
    correct_count = 0
    total_count = 0

    loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    with torch.no_grad():
        for idx, (image, label) in enumerate(loader):
            image = image.to(device)
            output = model(image)
            pred = output.argmax(dim=-1).squeeze(0)
            label = label.squeeze(0)

            pred_str = decode_prediction(pred.cpu())
            label_str = decode_prediction(label)

            total_count += 1
            if pred_str == label_str:
                correct_count += 1
            else:
                # Get original image path from dataset
                original_idx = val_dataset.indices[idx]
                img_path = full_dataset.image_paths[original_idx]
                errors.append({
                    "path": img_path,
                    "pred": pred_str,
                    "label": label_str,
                    "image": image.cpu().squeeze(0).squeeze(0).numpy(),
                })

    print(f"\nAccuracy: {correct_count}/{total_count} ({100*correct_count/total_count:.2f}%)")
    print(f"Errors: {len(errors)}")

    # Analyze error patterns
    print("\n=== Error Pattern Analysis ===")
    char_confusion = {}
    for err in errors:
        for i, (p, l) in enumerate(zip(err["pred"], err["label"])):
            if p != l:
                key = f"{l}->{p}"
                char_confusion[key] = char_confusion.get(key, 0) + 1

    print("\nTop 20 character confusions:")
    sorted_conf = sorted(char_confusion.items(), key=lambda x: -x[1])[:20]
    for conf, count in sorted_conf:
        print(f"  {conf}: {count}")

    # Save error images
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    num_to_save = min(num_samples, len(errors))
    cols = 5
    rows = (num_to_save + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
    axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes.flatten()

    for i, err in enumerate(errors[:num_to_save]):
        ax = axes[i]
        ax.imshow(err["image"], cmap="gray")
        ax.set_title(f"GT: {err['label']}\nPred: {err['pred']}", fontsize=10)
        ax.axis("off")

    # Hide unused subplots
    for i in range(num_to_save, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    save_path = output_path / "error_samples.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\nSaved error visualization to {save_path}")

    # Save detailed error list
    error_list_path = output_path / "error_list.txt"
    with open(error_list_path, "w") as f:
        f.write(f"Total errors: {len(errors)}/{total_count}\n")
        f.write(f"Accuracy: {100*correct_count/total_count:.2f}%\n\n")
        f.write("Top character confusions:\n")
        for conf, count in sorted_conf:
            f.write(f"  {conf}: {count}\n")
        f.write("\n\nAll errors:\n")
        for err in errors:
            f.write(f"{Path(err['path']).name}: GT={err['label']}, Pred={err['pred']}\n")
    print(f"Saved error list to {error_list_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze model errors")
    parser.add_argument(
        "--model",
        type=str,
        default="models/best_model.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="images",
        help="Path to images directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="error_analysis",
        help="Directory to save error analysis",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="Number of error samples to visualize",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Validation split ratio (must match training)",
    )
    args = parser.parse_args()

    analyze_errors(
        model_path=args.model,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        val_split=args.val_split,
    )


if __name__ == "__main__":
    main()
