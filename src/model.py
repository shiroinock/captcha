import torch
import torch.nn as nn


class CRNN(nn.Module):
    """CNN + BiLSTM for CAPTCHA recognition."""

    def __init__(self, num_chars: int = 5, num_classes: int = 62):
        super().__init__()
        self.num_chars = num_chars
        self.num_classes = num_classes

        self.cnn = nn.Sequential(
            # Block 1: (1, 64, 192) -> (64, 32, 96)
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 2: (64, 32, 96) -> (128, 16, 48)
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 3: (128, 16, 48) -> (256, 8, 48)
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),
            # Block 4: (256, 8, 48) -> (512, 4, 48)
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),
            # Block 5: (512, 4, 48) -> (512, 1, 48)
            nn.Conv2d(512, 512, (4, 1), padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.4,
        )

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CNN feature extraction
        x = self.cnn(x)  # (B, 512, 1, W)
        x = x.squeeze(2)  # (B, 512, W)
        x = x.permute(0, 2, 1)  # (B, W, 512)

        # LSTM sequence modeling
        x, _ = self.lstm(x)  # (B, W, 512)

        # Take num_chars evenly spaced positions
        B, W, C = x.shape
        indices = torch.linspace(0, W - 1, self.num_chars).long().to(x.device)
        x = x[:, indices, :]  # (B, num_chars, 512)

        # Classification
        x = self.dropout(x)
        x = self.fc(x)  # (B, num_chars, num_classes)
        return x


# Alias for compatibility
FastCaptchaSolver = CRNN
