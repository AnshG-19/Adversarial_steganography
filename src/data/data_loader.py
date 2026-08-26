import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class ImageStegoDataset(Dataset):
    def __init__(self, folder_path, img_size=(256, 256)):
        self.folder_path = folder_path
        self.image_files = [
            f for f in os.listdir(folder_path) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ] if os.path.exists(folder_path) else []
        
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.folder_path, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        return self.transform(image)


def get_dataloader(folder_path, batch_size=4, shuffle=True):
    dataset = ImageStegoDataset(folder_path=folder_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)