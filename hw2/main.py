import os
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from models import DCGANGenerator, DCGANDiscriminator
from datasets import FaceDataset
from tools import train_GAN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./hw2_data/face/')
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=100)
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
    train_set = FaceDataset(prefix=train_dir, trans=trans)
    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    gen = DCGANGenerator()
    dis = DCGANDiscriminator()
    # models = (DCGANGenerator(), DCGANDiscriminator())
    optimizers = (
        torch.optim.AdamW(gen.parameters(), lr=args.learning_rate),
        torch.optim.AdamW(dis.parameters(), lr=args.learning_rate)
    )
    criterions = (
        torch.nn.BCELoss(),
        torch.nn.BCELoss()
    )
    train_GAN(device, loader, (gen, dis), criterions, optimizers, args.num_epochs, args.batch_size)
    return


if __name__ == '__main__':
    main()
