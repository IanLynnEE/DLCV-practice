import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from my_datasets import part2_dataset

def gen_mean_std(dataset):
    dataloader = DataLoader(train_set, batch_size=16, shuffle=False)
    train = next(iter(dataloader))
    mean = np.mean(train.numpy(), axis=(0, 2, 3))
    std = np.std(train.numpy(), axis=(0, 2, 3))
    return mean, std

if __name__=='__main__':
    train_set = part2_dataset(prefix='./data/p2_data/train', trans=transforms.ToTensor())
    mean, std = gen_mean_std(train_set)
    print(mean, std)
