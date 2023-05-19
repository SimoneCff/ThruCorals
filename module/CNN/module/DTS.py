from __future__ import print_function, division

import os
from torch.utils.data import Dataset
from PIL import Image


class SeaDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        #Data
        self.data_dir = data_dir
        self.transform = transform

        #Class name & index
        self.class_name = os.listdir(data_dir)
        self.class_to_idx = {cname: idx for idx, cname in enumerate(self.class_name)}

        #List
        self.images = []
        self.labels = []

        #Populate Lists
        for cname in self.class_name:
            class_dir = os.path.join(data_dir, cname)
            for img_name in os.listdir(class_dir):
                if img_name.startswith("._") or img_name.startswith("."):
                    continue
                else: 
                    img_path = os.path.join(class_dir, img_name)
                    self.images.append(img_path)
                    self.labels.append(self.class_to_idx[cname])
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = self.images[index]
        label = self.labels[index]

        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label