# CAPTCHA OCR プロジェクト - 現状と今後の方針

## 現状のまとめ

### モデル性能

| 指標 | 値 |
|------|-----|
| Best val_seq_acc | 78-80% |
| Best val_char_acc | 91-92% |
| アーキテクチャ | CRNN (CNN + BiLSTM) |
| 入力 | Grayscale (1, 64, 192) |
| 出力 | 5文字 × 62クラス (a-z, A-Z, 0-9) |

### 精度頭打ちの原因

**データセットのラベルミス**が主因と判明。

- ファイル名（ラベル）と実際の画像で大文字・小文字が不一致
- モデルは62クラス（大文字小文字を区別）で学習している
- ラベルが間違っている以上、これ以上の精度向上は困難

### 試した対策

| 対策 | 効果 |
|------|------|
| Dropout強化 (0.3→0.5) | 過学習抑制に有効 |
| Label Smoothing (0.05→0.1) | 自信過剰を抑制 |
| 複合スコアによるEarly Stopping | val_loss上昇を防止 |
| Augmentation | CAPTCHA自体が歪んでいるため効果限定的 |

### 結論

モデル・学習手法は適切に機能している。精度向上にはデータ品質の改善が必要。

---

## 今後の方針

### 実用化に向けた対応

#### 1. Case-sensitivity の確認

対象CAPTCHAが大文字小文字を区別するか確認する。

- **case-insensitive の場合**: 小文字統一 (36クラス) で再学習
- **case-sensitive の場合**: データセットのラベル修正が必要

#### 2. 推論パイプラインの最適化

```
PyTorch (.pt) → ONNX (.onnx) → TensorRT (.engine)
```

- FP16量子化で推論速度向上
- RTX 2080 Ti でリアルタイム推論を目指す

---

## ブラウザ連携の実装案

ブラウザ上のCAPTCHAを自動解決するシステムの構成案。

### アーキテクチャ選択肢

#### A. ブラウザ拡張機能 + ローカルサーバー

```
┌─────────────────┐     HTTP/WebSocket     ┌─────────────────┐
│ Browser Extension│ ◄──────────────────► │ Local Server    │
│ (JavaScript)     │                       │ (Python/FastAPI)│
│                  │                       │                  │
│ - CAPTCHA検出    │                       │ - TensorRT推論  │
│ - 画像キャプチャ │                       │ - 結果返却      │
│ - 結果入力       │                       │                  │
└─────────────────┘                       └─────────────────┘
```

**メリット**:
- GPU推論が使える（高速）
- モデル更新が容易
- 拡張機能は軽量に保てる

**デメリット**:
- ローカルサーバーの起動が必要
- ポート開放のセキュリティ考慮

#### B. ブラウザ拡張機能 + ONNX Runtime Web

```
┌─────────────────────────────────────────┐
│ Browser Extension                        │
│ (JavaScript + ONNX Runtime Web)          │
│                                          │
│ - CAPTCHA検出                            │
│ - WebGL/WebGPU/WASM で推論              │
│ - 結果入力                               │
└─────────────────────────────────────────┘
```

**ONNX Runtime Web のバックエンド**:
- `webgpu` - 最新GPU API（最速、対応ブラウザ限定）
- `webgl` - WebGL経由でGPU推論（広くサポート）
- `wasm` - WebAssembly CPU推論（フォールバック）

```javascript
const session = await ort.InferenceSession.create('model.onnx', {
  executionProviders: ['webgl']  // or 'webgpu', 'wasm'
});
```

**メリット**:
- 完全にブラウザ内で完結
- 外部サーバー不要
- 配布が簡単
- WebGL/WebGPUでGPU推論可能

**デメリット**:
- ネイティブGPU (CUDA/TensorRT) よりは遅い
- モデルサイズに制限（拡張機能のサイズ上限）

#### C. 外部デスクトップアプリ

```
┌─────────────────────────────────────────┐
│ Desktop App (Electron / Tauri / Python) │
│                                          │
│ - 画面キャプチャ or ブラウザ自動操作    │
│ - TensorRT/ONNX推論                     │
│ - キー入力シミュレーション              │
└─────────────────────────────────────────┘
```

**メリット**:
- フルコントロール
- 任意のブラウザで動作
- GPU推論可能

**デメリット**:
- 画面キャプチャ方式は座標依存
- Selenium等は検出されやすい

### 推奨構成

**開発・実験段階**: A (拡張機能 + ローカルサーバー)
- 最も柔軟で、モデル改善のイテレーションが早い
- FastAPI + TensorRTで低レイテンシ推論

**配布・実用段階**: B (ONNX Runtime Web)
- ユーザーに追加ソフト導入を求めない
- ただし推論速度とモデルサイズの制約あり

---

## 次のステップ

1. [ ] 対象CAPTCHAのcase-sensitivityを確認
2. [ ] 必要に応じてデータ修正または小文字統一
3. [ ] ONNX/TensorRT変換スクリプト作成
4. [ ] ローカル推論サーバー (FastAPI) 作成
5. [ ] ブラウザ拡張機能のプロトタイプ作成

---

## ファイル構成

```
captcha/
├── src/
│   ├── model.py          # CRNNモデル定義
│   ├── dataset.py        # データセット・前処理
│   ├── train.py          # 学習スクリプト
│   └── analyze_errors.py # エラー分析ツール
├── models/
│   └── best_model.pt     # 学習済みモデル
├── runs/                 # 学習履歴
├── images/               # データセット (~113k images)
└── docs/
    └── STATUS.md         # このファイル
```
