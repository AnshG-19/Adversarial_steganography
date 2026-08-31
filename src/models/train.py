import torch
import torch.optim as optim
from src.models.steganography_net import StegoEncoder, StegoDecoder
from src.models.steganalysis_net import SteganalysisDetector
from src.attacks.adversarial_generator import generate_adversarial_stego
from src.utils.metrics import compute_psnr

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running pipeline on: {device}")

    # Initialize networks
    encoder = StegoEncoder().to(device)
    decoder = StegoDecoder().to(device)
    detector = SteganalysisDetector().to(device)

    # Dummy sample inputs for testing pipeline
    cover = torch.rand(1, 3, 256, 256).to(device)
    secret = torch.rand(1, 1, 256, 256).to(device)

    # 1. Embed Secret
    stego = encoder(cover, secret)
    psnr_val = compute_psnr(cover, stego)
    print(f"Stego Image Generated | PSNR: {psnr_val:.2f} dB")

    # 2. Adversarial Perturbation (fool detector into seeing Class 0: Cover)
    target_clean = torch.tensor([0], device=device)
    adv_stego = generate_adversarial_stego(detector, stego, target_clean, epsilon=0.02)
    print("Adversarial perturbation applied successfully.")

    # 3. Decode Secret
    extracted = decoder(adv_stego)
    print(f"Secret decoded with output shape: {extracted.shape}")

if __name__ == "__main__":
    main()
