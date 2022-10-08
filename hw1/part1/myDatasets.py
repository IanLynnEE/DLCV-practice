import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms, InterpolationMode


class part1_dataset(Dataset):
    def __init__(self, prefix, training=False):
        self.training = training
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
        means = [0.5076548, 0.48128527, 0.43116662]
        stds = [0.2627228, 0.25468898, 0.27363828]
        if self.training is True:
            trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(means, stds),
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(224, interpolation=InterpolationMode.BICUBIC),
            ])
        else:
            trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(means, stds),
                transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
            ])
        image = trans(image)
        return image, label_name
