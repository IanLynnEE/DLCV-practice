from torch.utils.data import DataLoader
import numpy as np

from myDatasets import part1_dataset


def get_mean_std(dataset):
    dataloader = DataLoader(train_set, batch_size=22500, shuffle=False)
    train = next(iter(dataloader))[0]
    mean = np.mean(train.numpy(), axis=(0, 2, 3))
    std = np.std(train.numpy(), axis=(0, 2, 3))
    return mean, std


if __name__ == '__main__':
    train_set = part1_dataset(prefix='./data/p1_data/train_50')
    mean, std = get_mean_std(train_set)
    print(mean, std)
