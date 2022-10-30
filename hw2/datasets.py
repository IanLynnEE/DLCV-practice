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


class DigitDataset(Dataset):
    def __init__(self, prefix, trans, labeled=False):
        self.trans = trans
        self.labeled = labeled
        # If the prefix is the label file (csv file), use it to load filenames.
        # Otherwise, load images from the directory.
        # If the prefix is not a label file, but labeled is set,
        # raise error. TODO
        if prefix.endswith('.csv'):
            df = pd.read_csv(prefix)
            self.images = [os.path.join(os.path.dirname(prefix), 'data', f) for f in df['image_name']]
            self.labels = df['label'].tolist()
        elif labeled is True:
            print('Error! Please provide a label file or unset `labeled`.')
        else:
            self.images = [os.path.join(prefix, f) for f in os.listdir(prefix) if f.endswith('.png')]
        print(f'Number of images is {len(self.images)}')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        image = self.trans(image)
        if self.labels is not None:
            return image, self.labels[idx]
        return image
