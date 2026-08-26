import torch
import torch.nn as nn

def fgsm_attack(image, epsilon, data_grad):
    """Fast Gradient Sign Method (FGSM) to fool steganalysis classifiers."""
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    return torch.clamp(perturbed_image, 0.0, 1.0)


def generate_adversarial_stego(detector_model, stego_image, target_label, epsilon=0.01):
    """
    Adds subtle adversarial noise to a stego image to make 
    the steganalysis detector classify it as a clean cover image (Class 0).
    """
    detector_model.eval()
    stego_image = stego_image.clone().detach().requires_grad_(True)
    
    output = detector_model(stego_image)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(output, target_label)
    
    detector_model.zero_grad()
    loss.backward()
    
    data_grad = stego_image.grad.data
    adv_stego = fgsm_attack(stego_image, epsilon, data_grad)
    return adv_stego.detach()