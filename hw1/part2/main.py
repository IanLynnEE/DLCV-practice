import os
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from my_models import VGG16_FCN32
from my_datasets import part2_dataset
from tools import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./data/p2_data/')
    parser.add_argument('--num_classes', type=int, default=7)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

    means = [0.40851322, 0.37851325, 0.28088534]
    stds = [0.14234462, 0.10848372, 0.09824718]
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(means, stds),
    ])
    train_dir = os.path.join(args.data_root, 'train')
    val_dir = os.path.join(args.data_root, 'validation')

    train_set = part2_dataset(prefix=train_dir, trans=trans, has_mask=True)
    val_set = part2_dataset(prefix=val_dir, trans=trans, has_mask=True)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    model = VGG16_FCN32(num_classes=args.num_classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    model.to(device)
    train(model, train_loader, val_loader, args.num_epochs, device, criterion, optimizer)
    return


if __name__ == '__main__':
    main()
