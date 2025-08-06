from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np

class HorseZebraDataset(Dataset):
    def __init__(self, root_zebra, root_horse, transform=None):
        self.root_zebra = root_zebra
        self.root_horse = root_horse
        self.transform = transform

        # Filter out Zone.Identifier files and only keep valid image files
        all_zebra_files = os.listdir(root_zebra)
        all_horse_files = os.listdir(root_horse)
        
        # Filter for valid image extensions and exclude Zone.Identifier files
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        self.zebra_images = [f for f in all_zebra_files 
                           if any(f.lower().endswith(ext) for ext in valid_extensions) 
                           and not f.endswith(':Zone.Identifier')]
        self.horse_images = [f for f in all_horse_files 
                           if any(f.lower().endswith(ext) for ext in valid_extensions) 
                           and not f.endswith(':Zone.Identifier')]
        
        self.length_dataset = max(len(self.zebra_images), len(self.horse_images)) # 1000, 1500
        self.zebra_len = len(self.zebra_images)
        self.horse_len = len(self.horse_images)
        
        print(f"Found {self.zebra_len} zebra images and {self.horse_len} horse images")

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        zebra_img = self.zebra_images[index % self.zebra_len]
        horse_img = self.horse_images[index % self.horse_len]

        zebra_path = os.path.join(self.root_zebra, zebra_img)
        horse_path = os.path.join(self.root_horse, horse_img)

        try:
            zebra_img = np.array(Image.open(zebra_path).convert("RGB"))
            horse_img = np.array(Image.open(horse_path).convert("RGB"))
        except Exception as e:
            print(f"Error loading images: {zebra_path}, {horse_path}")
            print(f"Error: {e}")
            # Return a black image as fallback
            zebra_img = np.zeros((256, 256, 3), dtype=np.uint8)
            horse_img = np.zeros((256, 256, 3), dtype=np.uint8)

        if self.transform:
            augmentations = self.transform(image=zebra_img, image0=horse_img)
            zebra_img = augmentations["image"]
            horse_img = augmentations["image0"]

        return zebra_img, horse_img




