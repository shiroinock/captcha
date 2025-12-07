# CAPTCHA OCR

PyTorch CRNN model for CAPTCHA recognition.

## Model

- Architecture: CNN + BiLSTM
- Input: Grayscale image (64x192)
- Output: 5 characters (a-z, A-Z, 0-9)
- Validation accuracy: ~80%

## Usage

```bash
# Setup
python -m venv .venv
source .venv/bin/activate
uv pip install -r pyproject.toml

# Train
python src/train.py --data-dir images

# Analyze errors
python src/analyze_errors.py --model runs/<run_dir>/best_model.pt
```

## Dataset

This project uses the [CAPTCHA Dataset](https://www.kaggle.com/datasets/parsasam/captcha-dataset) from Kaggle.

- License: [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/)
- ~113,000 CAPTCHA images
- 5 alphanumeric characters per image

## Project Structure

```
captcha/
├── src/
│   ├── model.py          # CRNN model
│   ├── dataset.py        # Dataset & transforms
│   ├── train.py          # Training script
│   └── analyze_errors.py # Error analysis
├── docs/
│   └── STATUS.md         # Project status & roadmap
└── runs/                 # Training artifacts
```

## License

MIT
