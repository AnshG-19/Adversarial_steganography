# Adversarial_steganography
this is a steganalysis repositry made to detect hidden info inside images and test accuracy by using adversarial methods 
Deep Learning for Steganalysis: CNN Detector & Adversarial RobustnessThis repository contains a PyTorch implementation of a Convolutional Neural Network (CNN) designed to detect image steganography (hidden data) at a challenging payload of 0.4 bits per pixel (bpp).Additionally, the project tests the robustness of the trained steganalysis model against Fast Gradient Sign Method (FGSM) adversarial attacks to demonstrate how easily steganalyzers can be fooled by adversarial perturbations.🚀 FeaturesCustom CNN Architecture: Built from scratch using PyTorch, featuring a frozen KV (High-Pass) Filter layer to extract spatial noise residuals.Steganalysis-Specific Optimizations:Utilizes the Adamax optimizer for stable convergence on high-frequency noise.Applies the Absolute Value Trick to noise residuals to prevent signal cancellation.Uses Un-squashed scaling and RandomCrop to preserve delicate $\pm 1$ pixel stego alterations (avoiding the smoothing effect of standard bilinear resizing).Adversarial Evaluation: Integrates torchattacks to automatically generate FGSM attacks at various $\epsilon$ (epsilon) budgets and plots the model's accuracy degradation.Comprehensive Metrics: Tracks Training/Validation Loss and Accuracy, Area Under the ROC Curve (AUC), and generates a Confusion Matrix.📁 Dataset StructureThis project uses the BOSSbase 256x256 dataset. The code expects the data to be structured into two main directories (Train and Test), each containing cover (clean) and stego (altered) subdirectories.Plaintextarchive/
├── boss_256_0.4/
│   ├── cover/       # 9,000 clean training images
│   └── stego/       # 9,000 stego training images
└── boss_256_0.4_test/
    ├── cover/       # 1,000 clean testing images
    └── stego/       # 1,000 stego testing images
(Update the TRAIN_DIR and TEST_DIR paths in the configuration section of the script to match your local machine).🛠️ Installation & RequirementsEnsure you have Python 3.8+ installed. It is highly recommended to use a virtual environment.1. Install PyTorch:Install PyTorch with CUDA support for your specific GPU. For Windows/CUDA 11.8:Bashpip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
2. Install dependencies:Bashpip install numpy pillow scikit-learn matplotlib torchattacks tqdm
💻 UsageRun the main script from your terminal:Bashpython steganalysis.py
What happens during execution:Training: The model trains for 15 epochs, displaying a live tqdm progress bar with Batch Loss, Training Accuracy, Validation Loss, and Validation Accuracy.Model Saving: The best weights are saved locally as stego_cnn.pt.Clean Evaluation: The model runs a final test on the unseen test set, printing the final AUC and Confusion Matrix.FGSM Attack: The script generates adversarial examples at $\epsilon = [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2]$ and outputs a line graph (accuracy_vs_epsilon.png) showing how the detector's accuracy drops as the attack strength increases.🧠 Architecture DetailsDetecting a 0.4 bpp payload is notoriously difficult because standard Convolutional Neural Networks tend to learn image content (edges, shapes) rather than the microscopic noise introduced by steganography.To force the CNN to look at the noise, the first layer is initialized with a Fixed KV Kernel and its gradients are frozen:PythonKV_KERNEL = torch.tensor([
    [-1,  2, -2,  2, -1],
    [ 2, -6,  8, -6,  2],
    [-2,  8, -12, 8, -2],
    [ 2, -6,  8, -6,  2],
    [-1,  2, -2,  2, -1],
], dtype=torch.float32) / 12.0
The network utilizes LeakyReLU activations to retain negative noise values and progressively downsamples the feature maps using AvgPool2d before a final fully connected classification head.📈 Future Work & Research AvenuesAdversarial Training: Modify the training loop to feed FGSM-perturbed images back into the model to build a robust defender.Advanced Architectures: Implement SRNet or Ye-Net architectures to push baseline accuracy higher on low-payload datasets.Modern Payloads: Test the model against newer steganographic algorithms like HILL or MiPOD.
Author
Ansh Gaur / (https://github.com/AnshG-19
