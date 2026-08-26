import torch
import torch.nn.functional as F

def compute_psnr(img1, img2):
    """Calculates Peak Signal-to-Noise Ratio (PSNR)."""
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float('inf')
    return (20 * torch.log10(1.0 / torch.sqrt(mse))).item()


def compute_ber(secret_true, secret_pred, threshold=0.5):
    """Calculates Bit Error Rate (BER) between binary payloads."""
    bin_true = (secret_true > threshold).float()
    bin_pred = (secret_pred > threshold).float()
    errors = torch.sum(torch.abs(bin_true - bin_pred))
    total_bits = torch.numel(bin_true)
    return (errors / total_bits).item()