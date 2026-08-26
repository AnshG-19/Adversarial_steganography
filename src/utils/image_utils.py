import matplotlib.pyplot as plt

def save_comparison_plot(cover, stego, adv_stego, save_path="comparison.png"):
    """Visualizes Cover vs Stego vs Adversarial Stego images."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    axes[0].imshow(cover.squeeze(0).permute(1, 2, 0).cpu().numpy())
    axes[0].set_title("Original Cover")
    axes[0].axis("off")

    axes[1].imshow(stego.squeeze(0).permute(1, 2, 0).cpu().numpy())
    axes[1].set_title("Stego Image")
    axes[1].axis("off")

    axes[2].imshow(adv_stego.squeeze(0).permute(1, 2, 0).cpu().numpy())
    axes[2].set_title("Adversarial Stego")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()