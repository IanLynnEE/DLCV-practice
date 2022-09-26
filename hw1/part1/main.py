import os
import argparse

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from myModels import myLeNet
from myDatasets import part1_dataset
from tools import train

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./data/p1_data/')
    parser.add_argument('--num_out', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=60470)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    device = 'mps'

    train_set = part1_dataset(prefix=os.path.join(args.data_root, 'train_50'))
    val_set = part1_dataset(prefix=os.path.join(args.data_root, 'val_50'))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)


    model = myLeNet(num_out=args.num_out)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer)
    criterion = torch.nn.CrossEntropyLoss()
    model.to(device)
    train(model, train_loader, val_loader, args.num_epochs, device, criterion, optimizer)
    return

if __name__ == '__main__':
    main()
