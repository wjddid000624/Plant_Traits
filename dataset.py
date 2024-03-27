import os
import csv
import torch
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

valid_images = [".jpg", ".gif", ".png", ".tga", ".jpeg", ".PNG", ".JPG", ".JPEG"]
with open("./class_info.json", 'r') as f:
    class2id = json.load(f)

class PlantDataset(Dataset):
    def __init__(
        self, 
        root: str, 
        split: str, 
        transforms=None
    ):
        self.root = root
        self.split = split
        self.transforms = transforms
        self.totensor = T.ToTensor()
        self.data = self.prepare_dataset()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        image = Image.open(self.data[index][0])
        if self.transforms:
            image = self.transforms(image)
        image = self.totensor(image)
        label = self.data[index][1]
        return {
            'input': image,
            'target': label 
        }
    
    def prepare_dataset(self):
        split_base = os.path.join(self.root, self.split)
        data = []
        
        for label in os.listdir(split_base):
            if label not in self.class2id:
                continue
            
            for image_name in os.listdir(os.path.join(split_base, label)):
                if os.path.splitext(image_name)[1] not in valid_images:
                    continue
                data.append((os.path.join(split_base, label, image_name), self.class2id[label]))
        
        return data