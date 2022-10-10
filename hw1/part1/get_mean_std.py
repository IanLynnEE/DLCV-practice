from torch.utils.data import DataLoader
import numpy as np

from myDatasets import part1_dataset


def get_mean_std(dataset):
    dataloader = DataLoader(train_set, batch_size=22500, shuffle=False)
    train = next(iter(dataloader))[0]
    mean = np.mean(train.numpy(), axis=(0, 2, 3))
    std = np.std(train.numpy(), axis=(0, 2, 3))
    return mean, std


def double_check():
    import pandas as pd
    df = pd.read_csv('val_gt.csv')

    count = 0
    data = df.to_numpy()
    for row in data:
        name = row[0]
        label = row[1]
        if int(name.split('_')[0]) == label:
            count += 1
    print(count, data.shape)

    df.info()


if __name__ == '__main__':
    # double_check()

    train_set = part1_dataset(prefix='./data/p1_data/train_50')
    mean, std = get_mean_std(train_set)
    print(mean, std)
