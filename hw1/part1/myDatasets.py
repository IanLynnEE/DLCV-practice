import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms

class part1_dataset(Dataset):
    def __init__(self, prefix):
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
        image = Image.open(img_name)
        means = [0.4822673, 0.44025022, 0.38372642]
        stds = [0.24469455, 0.23420024, 0.23852295]
        trans = transforms.Compose([
            transforms.Resize(224, interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(means, stds),
        ])
        image = trans(image)
        return image, label_name
