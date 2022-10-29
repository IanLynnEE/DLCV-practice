import os
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from models import DCGANGenerator, DCGANDiscriminator
from datasets import FaceDataset
from tools import train_GAN, load_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', type=str, default='./hw2_data/face/')
    parser.add_argument('--use_checkpoint', type=str, default='None')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0004)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

    if 'face' in args.prefix:
        train_GAN(device, *setup_DCGAN(args))
    return


def setup_DCGAN(args):
    # https://gist.github.com/weiaicunzai/e623931921efefd4c331622c344d8151
    means = [0.5071, 0.4865, 0.4409]
    stds = [0.2673, 0.2564, 0.2762]
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(means, stds),
        transforms.RandomHorizontalFlip(),
    ])
    train_dir = os.path.join(args.prefix, 'train')
    train_set = FaceDataset(prefix=train_dir, trans=trans)
    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    models = (
        DCGANGenerator(),
        DCGANDiscriminator()
    )
    criterions = (
        torch.nn.BCELoss(),
        torch.nn.BCELoss()
    )
    optimizers = (
        torch.optim.AdamW(models[0].parameters(), lr=args.lr),
        torch.optim.AdamW(models[1].parameters(), lr=args.lr)
    )
    epochs = range(args.epochs)
    if args.use_checkpoint == 'DCGAN':
        path = './saved_models/DCGANGenerator.pt'
        epoch = load_checkpoint(path, models[0], optimizers[0])
        path = './saved_models/DCGANDiscriminator.pt'
        if epoch != load_checkpoint(path, models[1], optimizers[1]):
            print('Warning: Using checkpoints from different epoch')
        epochs = range(epoch + 1, epoch + 1 + args.epochs)
    return loader, models, criterions, optimizers, epochs


if __name__ == '__main__':
    main()
