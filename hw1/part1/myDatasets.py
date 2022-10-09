import os

from PIL import Image
from torch.utils.data import Dataset


class part1_dataset(Dataset):
    def __init__(self, prefix, trans=None, has_label=False):
        self.prefix = prefix
        self.trans = trans
        self.has_label = has_label
        self.images = os.listdir(prefix)
        if has_label is True:
            self.labels = self.images.copy()
            for i, filename in enumerate(self.images):
                self.labels[i] = int(filename.split('_')[0])
        print(f'Number of images is {len(self.images)}')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.prefix, self.images[idx])
        image = Image.open(img_name).convert('RGB')
        image = self.trans(image)
        if self.has_label is True:
            label_name = self.labels[idx]
            return image, label_name
        return image, self.images[idx]
