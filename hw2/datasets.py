import os

from PIL import Image
import pandas as pd
from torch.utils.data import Dataset


class FaceDataset(Dataset):
    def __init__(self, prefix, trans):
        self.prefix = prefix
        self.trans = trans
        self.images = [os.path.join(prefix, f) for f in os.listdir(prefix) if f.endswith('.png')]
        print(f'Number of images is {len(self.images)}')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        image = self.trans(image)
        return image
