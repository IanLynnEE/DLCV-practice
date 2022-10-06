import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

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
            mask = (np.asarray(mask_image) > 127).astype(np.int64)
            mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]
            mask[mask == 3] = 0  # (Cyan: 011) Urban land 
            mask[mask == 6] = 1  # (Yellow: 110) Agriculture land 
            mask[mask == 5] = 2  # (Purple: 101) Rangeland 
            mask[mask == 2] = 3  # (Green: 010) Forest land 
            mask[mask == 1] = 4  # (Blue: 001) Water 
            mask[mask == 7] = 5  # (White: 111) Barren land 
            mask[mask == 0] = 6  # (Black: 000) Unknown
            mask[mask == 4] = 6  # Unknown
            return image, mask
        return image
