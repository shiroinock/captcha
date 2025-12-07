# 環境構築手順

## 前提条件

- Ubuntu (WSL2)
- NVIDIA GPU (RTX 2080 Ti)
- NVIDIA Driver 576.52+

## 1. uv のインストール

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

## 2. 仮想環境の作成

```bash
uv venv
source .venv/bin/activate
```

## 3. パッケージのインストール

`pyproject.toml` から一括インストール:

```bash
uv pip install -e .
```

ONNX エクスポート用パッケージも含める場合:

```bash
uv pip install -e ".[export]"
```

## 4. 動作確認

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

期待される出力:
```
CUDA: True NVIDIA GeForce RTX 2080 Ti
```

## インストール済みパッケージ

| パッケージ | バージョン |
|------------|------------|
| Python | 3.12.3 |
| uv | 0.9.16 |
| torch | 2.6.0+cu124 |
| torchvision | 0.21.0+cu124 |
| opencv-python | 4.12.0.88 |
| albumentations | 2.0.8 |
| onnx | 1.20.0 |
| onnxruntime-gpu | 1.23.2 |

## 学習の実行

```bash
# 基本的な学習（精度改善版のデフォルト設定）
python src/train.py --data-dir images

# 名前付きで実行（runs/YYYYMMDD_HHMMSS_experiment1/ に保存）
python src/train.py --data-dir images --name experiment1

# Focal Loss有効（混同しやすい文字の対策）
python src/train.py --data-dir images --focal-loss

# オプション一覧
# --epochs 100          エポック数（デフォルト: 100）
# --lr 3e-4             学習率（デフォルト: 3e-4）
# --weight-decay 1e-4   Weight Decay（デフォルト: 1e-4）
# --val-split 0.1       検証データの割合（デフォルト: 0.1）
# --save-dir models     モデル保存先（デフォルト: models）
# --label-smoothing 0.05 Label Smoothing（デフォルト: 0.05）
# --patience 15         Early Stopping patience（デフォルト: 15）
# --warmup-epochs 5     LR Warmupエポック数（デフォルト: 5）
# --focal-loss          Focal Lossを使用
# --focal-gamma 2.0     Focal Lossのgamma（デフォルト: 2.0）
# --run-dir runs        学習結果保存ディレクトリ（デフォルト: runs）
# --name NAME           実験名（ディレクトリ名のサフィックス）
```

### 学習結果の保存

各学習セッションは `runs/YYYYMMDD_HHMMSS/` に自動保存されます:

```
runs/
└── 20241215_143022/
    ├── config.json         # 学習設定（ハイパーパラメータ等）
    ├── history.csv         # エポックごとのメトリクス
    ├── training_curves.png # 学習曲線（Loss, Char Acc, Seq Acc）
    ├── results.json        # 最終結果サマリー
    ├── best_model.pt       # ベストモデル
    └── final_model.pt      # 最終エポックのモデル
```

これにより、バックグラウンド実行後にセッションを再起動しても結果が残ります。

## 推論

```bash
# 単一画像の推論
python src/inference.py images/1a6QN.jpg --model models/best_model.pt

# ディレクトリ内の画像を評価
python src/inference.py images --model models/best_model.pt

# TTA（Test Time Augmentation）で精度向上
python src/inference.py images --model models/best_model.pt --tta

# ベンチマーク（推論速度計測）
python src/inference.py images --model models/best_model.pt --benchmark
```

## モデルエクスポート

```bash
# ONNXエクスポート（simplify無効）
python src/export.py --model-path models/best_model.pt --format onnx --no-simplify

# ONNX推論
python src/inference.py images/1a6QN.jpg --model models/model.onnx
```

## 推論速度の最適化

### torch.compile（推奨）

TensorRTがインストールされていない環境では、`torch.compile`が最も簡単な高速化手段です。

```python
import torch
from src.model import FastCaptchaSolver

model = FastCaptchaSolver().cuda().eval()
model.load_state_dict(torch.load('models/best_model.pt', weights_only=True))

# コンパイル（初回実行時に最適化）
model_compiled = torch.compile(model, mode='reduce-overhead')
```

### 推論速度比較（RTX 2080 Ti）

| 形式 | 平均 | 最小 |
|------|------|------|
| PyTorch FP32 | 5.5 ms | 3.3 ms |
| ONNX (CPU) | 5.3 ms | 4.4 ms |
| ONNX (CUDA) | 3.8 ms | 2.3 ms |
| torch.compile | 2.9 ms | 1.9 ms |
| TensorRT FP16 | 0.5-1 ms | - |

※ ONNX (CUDA) には onnxruntime-gpu と CUDA 12 が必要
※ TensorRTはpip版だけでは完全動作せず、NVIDIA公式CUDA Toolkitが必要

### TensorRTの問題点

pip版TensorRT (`tensorrt`, `tensorrt-cu13-libs`) をインストールしても、onnxruntimeのTensorrtExecutionProviderは以下のエラーで動作しない:

```
libcurand.so.10: cannot open shared object file
libnvinfer.so.10: cannot open shared object file
```

**推奨**: torch.compileで2ms未満を達成できるため、TensorRTは不要。
どうしてもTensorRTを使いたい場合は、NVIDIA公式サイトからCUDA ToolkitとTensorRTをシステムにインストールする必要がある。

## 現在の学習結果

| 指標 | 値 |
|------|-----|
| クラス数 | 62（a-z, A-Z, 0-9） |
| Best Val Seq Acc | 80.3% |
| Best Val Char Acc | 92.3% |
| モデル | CRNN (CNN + BiLSTM) |
| Label Smoothing | 0.1 |
| エポック数 | 50 |

## 精度改善（v2）

以下の改善を実装済み:

### 学習の安定化
- **Early Stopping**: patience=15 で過学習を防止
- **Gradient Clipping**: max_norm=5.0 でLSTMの勾配爆発を防止
- **LR Warmup**: 5エポックのウォームアップ後にCosine Annealing

### ハイパーパラメータ調整
- エポック数: 30 → 100
- 学習率: 1e-3 → 3e-4（より安定）
- Label Smoothing: 0.1 → 0.05（識別性向上）

### Data Augmentation調整
- ノイズ・ブラー・ドロップアウトを控えめに調整
- CAPTCHAは元々歪んでいるため、過度な拡張は逆効果

### Focal Loss（オプション）
- `--focal-loss` で有効化
- O/0, l/1/I など混同しやすい文字のhard exampleに重点

### TTA（推論時）
- `--tta` で有効化
- 明るさを変えた3パターンの確率平均で予測
