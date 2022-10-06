import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms

class part2_dataset(Dataset):
    def __init__(self, prefix, trans, has_mask=False):
        self.prefix = prefix
        self.images = [name for name in os.listdir(prefix) if 'sat' in name]
        self.trans = trans
        self.has_mask = has_mask
        print(f'Number of images is {len(self.images)}')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_filename = os.path.join(self.prefix, self.images[idx])
        image = Image.open(img_filename)
        image = self.trans(image)
        if self.has_mask:
            mask_filename = img_filename.replace('_sat.jpg', '_mask.png')
            mask_image = Image.open(mask_filename)
            mask = (np.asarray(mask_image) == 255).astype(np.int64)
            mask_label = 3 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]
            return image, mask_label
        return image
