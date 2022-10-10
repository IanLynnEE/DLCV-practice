import os

from PIL import Image
import numpy as np
from torch.utils.data import Dataset


def read_masks(prefix, file_list):
    masks = np.empty((len(file_list), 512, 512), np.int_)

    for i, file in enumerate(file_list):
        # Pillow is much faster than imageio.
        mask = Image.open(os.path.join(prefix, file)).convert('RGB')
        mask = (np.asarray(mask) > 127).astype(np.int_)
        # As the dataset does not use Red: 100, it's okay to use:
        # mask = 3 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]
        # But modifications are necessary for viz_mask.py by doing so.
        mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]
        masks[i, mask == 3] = 0     # (Cyan: 011) Urban land
        masks[i, mask == 6] = 1     # (Yellow: 110) Agriculture land
        masks[i, mask == 5] = 2     # (Purple: 101) Rangeland
        masks[i, mask == 2] = 3     # (Green: 010) Forest land
        masks[i, mask == 1] = 4     # (Blue: 001) Water
        masks[i, mask == 7] = 5     # (White: 111) Barren land
        masks[i, mask == 0] = 6     # (Black: 000) Unknown
        masks[i, mask == 4] = 6     # Undefined
    return masks


class part2_dataset(Dataset):
    def __init__(self, prefix, trans, has_mask=False):
        self.prefix = prefix
        self.trans = trans
        self.has_mask = has_mask
        self.images = [f for f in os.listdir(prefix) if f.endswith('.jpg')]
        if has_mask is True:
            self.masks = read_masks(prefix, self.images)  # RAM consuming but faster.
        print(f'Number of images is {len(self.images)}')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.prefix, self.images[idx])
        image = Image.open(image_path).convert('RGB')
        image = self.trans(image)
        if self.has_mask is True:
            return image, self.masks[idx], self.images[idx]
        return image, self.images[idx]
