import argparse
from pathlib import Path

import torch


def export_onnx(
    model_path: str | Path,
    output_path: str | Path,
    num_chars: int = 5,
    num_classes: int = 62,
    simplify: bool = True,
) -> None:
    from model import FastCaptchaSolver

    model = FastCaptchaSolver(num_chars=num_chars, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()

    dummy_input = torch.randn(1, 1, 64, 192)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )
    print(f"Exported ONNX model to {output_path}")

    if simplify:
        import onnx
        from onnxsim import simplify as onnx_simplify

        model_onnx = onnx.load(str(output_path))
        model_onnx, check = onnx_simplify(model_onnx)
        if check:
            onnx.save(model_onnx, str(output_path))
            print("Simplified ONNX model")
        else:
            print("Warning: ONNX simplification failed")


def export_tensorrt(
    model_path: str | Path,
    output_path: str | Path,
    num_chars: int = 5,
    num_classes: int = 62,
    fp16: bool = True,
) -> None:
    from torch2trt import torch2trt

    from model import FastCaptchaSolver

    model = FastCaptchaSolver(num_chars=num_chars, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model = model.cuda().eval()

    dummy_input = torch.randn(1, 1, 64, 192).cuda()

    model_trt = torch2trt(model, [dummy_input], fp16_mode=fp16)
    torch.save(model_trt.state_dict(), str(output_path))
    print(f"Exported TensorRT model to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Export CAPTCHA model")
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/best_model.pt",
        help="Path to PyTorch model",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Output directory",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["onnx", "tensorrt", "both"],
        default="onnx",
        help="Export format",
    )
    parser.add_argument(
        "--no-simplify",
        action="store_true",
        help="Skip ONNX simplification",
    )
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="Use FP32 for TensorRT (default: FP16)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    if args.format in ["onnx", "both"]:
        export_onnx(
            args.model_path,
            output_dir / "model.onnx",
            simplify=not args.no_simplify,
        )

    if args.format in ["tensorrt", "both"]:
        export_tensorrt(
            args.model_path,
            output_dir / "model.trt",
            fp16=not args.fp32,
        )


if __name__ == "__main__":
    main()
