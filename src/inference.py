import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from dataset import CHARS, IDX_TO_CHAR


# TTA transforms (brightness variations only - no flips for text)
TTA_TRANSFORMS = [
    lambda x: x,  # Original
    lambda x: np.clip(x * 1.1, 0, 255).astype(np.uint8),  # Brighter
    lambda x: np.clip(x * 0.9, 0, 255).astype(np.uint8),  # Darker
]


class FastPreprocessor:
    def __init__(self, height: int = 64, width: int = 192):
        self.height = height
        self.width = width

    def __call__(self, image_path: str | Path) -> np.ndarray:
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        img = cv2.resize(img, (self.width, self.height))
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        return img[np.newaxis, np.newaxis, :, :]


class PyTorchInference:
    def __init__(self, model_path: str | Path, device: str = "cuda", use_tta: bool = False):
        from model import FastCaptchaSolver

        self.device = torch.device(device)
        self.model = FastCaptchaSolver()
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True)
        )
        self.model = self.model.to(self.device).eval()
        self.preprocessor = FastPreprocessor()
        self.use_tta = use_tta

    def _preprocess_raw(self, img: np.ndarray) -> np.ndarray:
        """Preprocess a raw grayscale image (already resized)."""
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        return img[np.newaxis, np.newaxis, :, :]

    @torch.no_grad()
    def predict(self, image_path: str | Path) -> str:
        if self.use_tta:
            return self.predict_tta(image_path)

        input_data = self.preprocessor(image_path)
        input_tensor = torch.from_numpy(input_data).to(self.device)
        output = self.model(input_tensor)
        indices = output[0].argmax(dim=-1)
        return "".join(IDX_TO_CHAR[i.item()] for i in indices)

    @torch.no_grad()
    def predict_tta(self, image_path: str | Path) -> str:
        """Predict using Test Time Augmentation (average probabilities)."""
        # Load and resize image once
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        img = cv2.resize(img, (self.preprocessor.width, self.preprocessor.height))

        all_probs = []
        for transform in TTA_TRANSFORMS:
            aug_img = transform(img.copy())
            input_data = self._preprocess_raw(aug_img)
            input_tensor = torch.from_numpy(input_data).to(self.device)
            output = self.model(input_tensor)
            probs = F.softmax(output[0], dim=-1)
            all_probs.append(probs)

        # Average probabilities across TTA transforms
        avg_probs = torch.stack(all_probs).mean(dim=0)
        indices = avg_probs.argmax(dim=-1)
        return "".join(IDX_TO_CHAR[i.item()] for i in indices)

    @torch.no_grad()
    def predict_batch(self, image_paths: list[str | Path]) -> list[str]:
        inputs = np.concatenate(
            [self.preprocessor(p) for p in image_paths], axis=0
        )
        input_tensor = torch.from_numpy(inputs).to(self.device)
        outputs = self.model(input_tensor)
        results = []
        for output in outputs:
            indices = output.argmax(dim=-1)
            results.append("".join(IDX_TO_CHAR[i.item()] for i in indices))
        return results


class ONNXInference:
    def __init__(self, model_path: str | Path, device: str = "cuda"):
        import onnxruntime as ort

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device == "cuda"
            else ["CPUExecutionProvider"]
        )
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.preprocessor = FastPreprocessor()

    def predict(self, image_path: str | Path) -> str:
        input_data = self.preprocessor(image_path)
        output = self.session.run(None, {self.input_name: input_data})[0]
        indices = output[0].argmax(axis=-1)
        return "".join(IDX_TO_CHAR[i] for i in indices)


def benchmark(
    inference_fn,
    image_paths: list[Path],
    warmup: int = 10,
    iterations: int = 100,
) -> dict:
    for _ in range(warmup):
        inference_fn(image_paths[0])

    times = []
    for i in range(iterations):
        path = image_paths[i % len(image_paths)]
        start = time.perf_counter()
        inference_fn(path)
        end = time.perf_counter()
        times.append((end - start) * 1000)

    return {
        "mean_ms": np.mean(times),
        "std_ms": np.std(times),
        "min_ms": np.min(times),
        "max_ms": np.max(times),
    }


def main():
    parser = argparse.ArgumentParser(description="CAPTCHA inference")
    parser.add_argument("image", type=str, help="Image path or directory")
    parser.add_argument(
        "--model",
        type=str,
        default="models/best_model.pt",
        help="Model path (.pt or .onnx)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark",
    )
    parser.add_argument(
        "--tta",
        action="store_true",
        help="Use Test Time Augmentation for improved accuracy",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    if model_path.suffix == ".onnx":
        engine = ONNXInference(model_path, args.device)
    else:
        engine = PyTorchInference(model_path, args.device, use_tta=args.tta)
        if args.tta:
            print("TTA enabled")

    image_path = Path(args.image)

    if image_path.is_dir():
        image_paths = list(image_path.glob("*.jpg")) + list(image_path.glob("*.png"))
        if args.benchmark:
            results = benchmark(engine.predict, image_paths)
            print(f"Benchmark results:")
            print(f"  Mean: {results['mean_ms']:.3f} ms")
            print(f"  Std:  {results['std_ms']:.3f} ms")
            print(f"  Min:  {results['min_ms']:.3f} ms")
            print(f"  Max:  {results['max_ms']:.3f} ms")
        else:
            correct = 0
            for p in image_paths[:100]:
                pred = engine.predict(p)
                label = p.stem
                if pred == label:
                    correct += 1
                else:
                    print(f"{p.name}: pred={pred}, label={label}")
            print(f"Accuracy: {correct}/{min(100, len(image_paths))}")
    else:
        pred = engine.predict(image_path)
        print(f"Prediction: {pred}")


if __name__ == "__main__":
    main()
