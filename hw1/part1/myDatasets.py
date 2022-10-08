import os

from PIL import Image
from torch.utils.data import Dataset


class part1_dataset(Dataset):
    def __init__(self, prefix, trans=None):
        self.trans = trans
        self.images = os.listdir(prefix)
        self.labels = self.images.copy()
        for i, filename in enumerate(self.images):
            self.labels[i] = int(filename.split('_')[0])
        self.prefix = prefix
        print(f'Number of images is {len(self.images)}')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.prefix, self.images[idx])
        label_name = self.labels[idx]
        image = Image.open(img_name).convert('RGB')
        image = self.trans(image)
        return image, label_name
