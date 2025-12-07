# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CAPTCHA OCR model for recognizing distorted alphanumeric strings. The goal is real-time inference with low latency on RTX 2080 Ti.

## Development Commands

```bash
# Activate environment
source .venv/bin/activate

# Verify GPU
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# Install packages (use uv, not pip)
uv pip install <package>
```

## Architecture

See `plan.md` for detailed design specifications.

**Model**: `FastCaptchaSolver` - Lightweight CNN with per-position classification heads
- Input: Grayscale (1, 64, 192)
- Output: (num_chars, num_classes) - 5 characters, 36 classes (a-z + 0-9)
- No CTC (fixed-length CAPTCHA)

**Optimization pipeline**: PyTorch FP32 → ONNX → TensorRT FP16

## Data

- `images/`: ~113k CAPTCHA images
- Filename = label (e.g., `11J8.jpg` → label is `11J8`)
- 5 characters per image, case-insensitive

## Key Design Decisions

- Grayscale input to reduce channels
- No BatchNorm (inference overhead)
- stride=2 convolutions instead of MaxPool
- Single Linear classifier (faster than ModuleList)