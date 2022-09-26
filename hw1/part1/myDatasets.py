import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms

class part1_dataset(Dataset):
    def __init__(self, prefix):
        self.images = os.listdir(prefix)
        print(type(self.images))
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
        # https://pytorch.org/vision/stable/transforms.html
        means = [0.485, 0.456, 0.406]
        stds = [0.229, 0.224, 0.225]
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(means, stds),
        ])
        image = trans(image)
        return image, label_name
