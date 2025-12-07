# CAPTCHA OCR モデル設計仕様

## 概要

歪んだアルファベット文字列を含む CAPTCHA 画像を認識するモデルを構築する。

## 要件

- **タスク**: 固定長 CAPTCHA 画像の文字認識
- **データセット**: Kaggle CAPTCHA 系、10 万枚以上
- **重要要件**: リアルタイム推論（低レイテンシ最優先）
- **開発環境**: Ubuntu + RTX 2080 Ti

## 技術スタック

### フレームワーク・ライブラリ

```
pytorch >= 2.0
torchvision
opencv-python      # 高速前処理
albumentations     # データ拡張
onnx
onnxsim
tensorrt           # 推論最適化
torch2trt          # TensorRT変換の簡易化（オプション）
wandb              # 実験管理（オプション）
```

### アーキテクチャ

固定長 CAPTCHA のため、CTC 不要。CNN + 各位置ごとの分類ヘッド構成。

```
入力: グレースケール画像 (1, 64, 192)
    ↓
Conv2d(1→32, stride=2) + ReLU
    ↓
Conv2d(32→64, stride=2) + ReLU
    ↓
Conv2d(64→128, stride=2) + ReLU
    ↓
AdaptiveAvgPool2d → Flatten
    ↓
Linear(128 → num_chars × num_classes)
    ↓
出力: (num_chars, num_classes)
```

設計ポイント:

- グレースケール入力でチャンネル削減
- BatchNorm 削除（推論オーバーヘッド削減）
- stride=2 の畳み込み（MaxPool より高速）
- 分類ヘッド統合（ModuleList より単一 Linear が高速）

### モデル実装

```python
import torch
import torch.nn as nn

class FastCaptchaSolver(nn.Module):
    def __init__(self, num_chars=5, num_classes=36):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(128, num_chars * num_classes)
        self.num_chars = num_chars
        self.num_classes = num_classes

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x.view(-1, self.num_chars, self.num_classes)
```

## 学習設定

### ハイパーパラメータ

| 項目           | 値                |
| -------------- | ----------------- |
| オプティマイザ | AdamW             |
| 学習率         | 1e-3              |
| Weight Decay   | 1e-4              |
| スケジューラ   | CosineAnnealingLR |
| バッチサイズ   | 64〜128           |
| エポック数     | 20〜30            |

### 損失関数

各位置ごとの CrossEntropyLoss の合計:

```python
loss = sum(
    F.cross_entropy(outputs[:, i, :], labels[:, i])
    for i in range(num_chars)
)
```

### データ拡張（控えめに）

CAPTCHA は既に歪んでいるため過度な拡張は逆効果:

```python
train_transforms = A.Compose([
    A.Resize(64, 192),
    A.GaussNoise(var_limit=(5, 20), p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
    A.Normalize(mean=0.5, std=0.5),
    ToTensorV2(),
])
```

### ラベル処理

```python
CHARS = 'abcdefghijklmnopqrstuvwxyz0123456789'
CHAR_TO_IDX = {c: i for i, c in enumerate(CHARS)}

def encode_label(text):
    return [CHAR_TO_IDX[c.lower()] for c in text]

def decode_prediction(outputs):
    indices = outputs.argmax(dim=-1)
    return ''.join(CHARS[i] for i in indices)
```

## 推論最適化

### パイプライン

```
PyTorch (FP32) → ONNX → TensorRT (FP16)
```

### ONNX エクスポート

```python
def export_onnx(model, save_path="model.onnx"):
    model.eval()
    dummy = torch.randn(1, 1, 64, 192)
    torch.onnx.export(
        model, dummy, save_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
    )

    # 最適化
    import onnx
    from onnxsim import simplify
    model_onnx = onnx.load(save_path)
    model_onnx, _ = simplify(model_onnx)
    onnx.save(model_onnx, save_path)
```

### 高速前処理

```python
import cv2
import numpy as np

class FastPreprocessor:
    def __init__(self, height=64, width=192):
        self.height = height
        self.width = width

    def __call__(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (self.width, self.height))
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        return img[np.newaxis, np.newaxis, :, :]
```

### TensorRT 変換（torch2trt 利用）

```python
from torch2trt import torch2trt

model = FastCaptchaSolver().cuda().eval()
dummy = torch.randn(1, 1, 64, 192).cuda()
model_trt = torch2trt(model, [dummy], fp16_mode=True)
```

## 期待性能

### 精度

- 文字単位: 98〜99%以上
- 全体正解率: 90〜95%以上

### レイテンシ（2080 Ti）

| 構成           | 推論時間    |
| -------------- | ----------- |
| PyTorch FP32   | 2〜3 ms     |
| PyTorch FP16   | 1〜2 ms     |
| TensorRT FP16  | 0.3〜0.8 ms |
| 前処理込み合計 | 1〜2 ms     |

## 開発フロー

1. データ準備（画像パス + ラベルの CSV/JSON 作成）
2. Dataset/DataLoader 実装
3. PyTorch で学習
4. 精度評価（char_accuracy, sequence_accuracy）
5. ONNX エクスポート
6. TensorRT 変換
7. ベンチマーク・デプロイ

## ディレクトリ構成（案）

```
captcha-ocr/
├── data/
│   ├── train/
│   ├── val/
│   └── labels.csv
├── src/
│   ├── model.py
│   ├── dataset.py
│   ├── train.py
│   ├── export.py
│   └── inference.py
├── models/
│   ├── model.pt
│   ├── model.onnx
│   └── model.trt
└── requirements.txt
```

## 進捗状況

### 完了

- [x] データ準備（画像パス + ラベル）
  - `images/` に 113,062 枚の CAPTCHA 画像
  - ファイル名がラベル（5文字、例: `11J8.jpg`）
- [x] Dataset/DataLoader 実装（`src/dataset.py`）
  - `CaptchaDataset`: 画像読み込み + ラベルエンコード
  - `get_train_transforms()`: GaussNoise, RandomBrightnessContrast
  - `get_val_transforms()`: リサイズ + 正規化のみ
- [x] モデル実装（`src/model.py`）
  - `FastCaptchaSolver`: 3層CNN + 単一Linear分類器
- [x] 学習スクリプト（`src/train.py`）
  - AdamW + CosineAnnealingLR
  - char_accuracy, sequence_accuracy の評価
  - ベストモデル保存
- [x] エクスポート（`src/export.py`）
  - ONNX エクスポート + onnxsim 最適化
  - TensorRT エクスポート（torch2trt）
- [x] 推論スクリプト（`src/inference.py`）
  - PyTorch / ONNX 推論
  - ベンチマーク機能

### 未完了

- [ ] 精度評価（目標: char 98%+, seq 90%+）
- [ ] ONNX エクスポート実行
- [ ] TensorRT 変換実行
- [ ] ベンチマーク実行（目標: < 2ms）

## 精度向上のための改善案

現在の課題: 大文字小文字区別（62クラス）で Seq Acc 78% 程度

### 優先度高（実装が簡単で効果的）

1. **Label Smoothing**
   - CrossEntropyLoss に `label_smoothing=0.1` を追加
   - 過学習抑制、汎化性能向上

2. **Test Time Augmentation (TTA)**
   - 推論時に元画像 + 左右反転 + 輝度変更などで複数予測
   - 多数決または確率平均で最終予測

3. **学習エポック数増加**
   - 30 → 50〜100 エポック
   - Early Stopping と組み合わせ

### 優先度中（効果的だが実装に時間がかかる）

4. **Attention機構追加**
   - 各文字位置に注目させるSelf-Attention
   - Transformer Encoder の導入

5. **Hard Example Mining**
   - 間違えやすいサンプルを重点学習
   - 損失が大きいサンプルの重み付け

6. **類似文字対策**
   - O/0, l/1/I, S/5 などの混同対策
   - Case-aware loss（大文字小文字混同にペナルティ）

### 優先度低（最終手段）

7. **アンサンブル**
   - 異なるseedで学習した複数モデルの投票
   - 精度向上するが推論時間増加

8. **より深いモデル**
   - ResNetブロック、SEブロック導入
   - レイテンシとのトレードオフ

### 実装済み対策

- [x] Data Augmentation強化（GaussNoise, Blur, Affine, CoarseDropout）
- [x] Dropout追加（LSTM 0.3, FC前 0.3）

### 実行コマンド

```bash
# 環境有効化
source .venv/bin/activate

# 学習
python src/train.py --data-dir images --epochs 30 --batch-size 128

# ONNXエクスポート
python src/export.py --model-path models/best_model.pt --format onnx

# 推論テスト
python src/inference.py images/11J8.jpg --model models/best_model.pt

# ベンチマーク
python src/inference.py images --model models/best_model.pt --benchmark
```
