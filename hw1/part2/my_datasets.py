import os

import imageio
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


def read_masks(filepath):
    file_list = [file for file in os.listdir(filepath) if file.endswith('.png')]
    file_list.sort()
    masks = np.empty((len(file_list), 512, 512), np.int_)

    for i, file in enumerate(file_list):
        mask = imageio.imread(os.path.join(filepath, file))
        mask = (mask > 127).astype(np.int_)
        mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]
        masks[i, mask == 3] = 0  # (Cyan: 011) Urban land
        masks[i, mask == 6] = 1  # (Yellow: 110) Agriculture land
        masks[i, mask == 5] = 2  # (Purple: 101) Rangeland
        masks[i, mask == 2] = 3  # (Green: 010) Forest land
        masks[i, mask == 1] = 4  # (Blue: 001) Water
        masks[i, mask == 7] = 5  # (White: 111) Barren land
        masks[i, mask == 0] = 6  # (Black: 000) Unknown
        masks[i, mask == 4] = 6  # Undefined
    return masks, file_list


class part2_dataset(Dataset):
    def __init__(self, prefix, trans, has_mask=False):
        self.prefix = prefix
        self.trans = trans
        self.has_mask = has_mask
        self.masks, files = read_masks(prefix)
        self.images = [os.path.join(prefix, file.replace('mask.png', 'sat.jpg')) for file in files]
        print(f'Number of images is {len(self.images)}')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        image = self.trans(image)
        if self.has_mask:
            return image, self.masks[idx]
        return image