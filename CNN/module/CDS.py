from __future__ import print_function, division
import os
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

def convert_to_vector(classes):
    tmp = []
    data_df = pd.read_csv(classes)
    for index, row in data_df.iterrows():
        tmp.append(row['class_name'])
    return tmp

class CoralDataset(Dataset):
    def __init__(self, csv, classes ,data_dir , scale ,transform=None):
        self.data_dir = data_dir
        self.csv_file = csv
        self.transform = transform
        self.scale = scale
        self.data_df = pd.read_csv(csv)

        self.class_name = convert_to_vector(classes)
        self.class_to_idx = {cname: idx for idx, cname in enumerate(self.class_name)}
       
        self.images = []
        self.label = []
        self.rows = []
        self.cols = []

        for index, row in self.data_df.iterrows():
            cname = row['Label']
            img_name = row['Name']
            if img_name.startswith("._") or img_name.startswith("."):
                continue
            else:
                self.label.append(self.class_to_idx[cname]) 
                img_path = os.path.join(data_dir, img_name)
                self.images.append(img_path)
                self.rows.append(row['Row'])
                self.cols.append(row['Column'])
                
                

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = self.images[index]
        label = self.label[index]
        row = self.rows[index]
        col = self.cols[index]

        img = Image.open(img_path).convert('RGB')
        #cropping image
        img_width = img.width
        img_height = img.height
        half_width = int(img_width * self.scale)
        half_height = int(img_height * self.scale)
        
        #Calculate x and y
        x_min = max(0,col - half_width)
        y_min = max(0,row - half_height)

        x_max = min(img_width,col + half_width)
        y_max = min(img_height,row + half_height)
        
        img = img.crop((x_min, y_min, x_max, y_max))

        if self.transform is not None:
            img = self.transform(img)

        return img, label