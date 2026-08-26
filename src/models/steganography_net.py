import torch
import torch.nn as nn

class StegoEncoder(nn.Module):
    """Encodes a secret message into a cover image."""
    def __init__(self, in_channels=3, secret_channels=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + secret_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, in_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, cover, secret):
        x = torch.cat([cover, secret], dim=1)
        residual = self.conv(x)
        stego_image = torch.clamp(cover + residual * 0.1, 0.0, 1.0)
        return stego_image


class StegoDecoder(nn.Module):
    """Extracts the secret message from a stego image."""
    def __init__(self, in_channels=3, secret_channels=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, secret_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, stego_image):
        return self.conv(stego_image)