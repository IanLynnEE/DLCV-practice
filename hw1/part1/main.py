import os
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms, InterpolationMode
import matplotlib.pyplot as plt

from myModels import myLeNet, myResNet50
from myDatasets import part1_dataset
from tools import train


def fix_seed():
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./data/p1_data/')
    parser.add_argument('--num_out', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=60470)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

    train_path = os.path.join(args.data_root, 'train_50')
    valid_path = os.path.join(args.data_root, 'val_50')
    means = [0.5076548, 0.48128527, 0.43116662]
    stds = [0.2627228, 0.25468898, 0.27363828]
    train_trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(means, stds),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(224, interpolation=InterpolationMode.BICUBIC),
    ])
    valid_trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(means, stds),
        transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
    ])

    model = myResNet50(num_out=args.num_out)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    train_set = part1_dataset(prefix=train_path, trans=train_trans)
    valid_set = part1_dataset(prefix=valid_path, trans=valid_trans)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(valid_set, batch_size=128, shuffle=False)
    model.to(device)
    train(model, train_loader, val_loader, args.num_epochs, device, criterion, optimizer)
    return


if __name__ == '__main__':
    main()
