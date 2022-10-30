import os
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from models import DCGANGenerator, DCGANDiscriminator
from models import DANNFeature, DANNLabel, DANNDomain
from datasets import FaceDataset, DigitDataset
from tools import train_DCGAN, train_DANN, load_checkpoint


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
        train_DCGAN(device, *setup_DCGAN(args))
    elif 'usps' in args.prefix:
        source = 'hw2_data/digits/mnistm/train.csv'
        target = 'hw2_data/digits/usps/train.csv'
        valid = 'hw2_data/digits/usps/val.csv'
        train_DANN(device, *setup_DANN(args, source, target, valid), weight=0.05)
    elif 'svhn' in args.prefix:
        source = 'hw2_data/digits/mnistm/train.csv'
        target = 'hw2_data/digits/svhn/train.csv'
        valid = 'hw2_data/digits/svhn/val.csv'
        train_DANN(device, *setup_DANN(args, source, target, valid), weight=0.05)
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


def setup_DANN(args, source_path, target_path, valid_path):
    # TODO: Use get_mean_std() to correct here.
    means = [0.5071, 0.4865, 0.4409]
    stds = [0.2673, 0.2564, 0.2762]
    trans = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(means, stds),
    ])
    source_set = DigitDataset(prefix=source_path, trans=trans, labeled=True)
    target_set = DigitDataset(prefix=target_path, trans=trans, labeled=False)
    valid_set = DigitDataset(prefix=valid_path, trans=trans, labeled=True)
    source = DataLoader(source_set, batch_size=args.batch_size, shuffle=True)
    target = DataLoader(target_set, batch_size=args.batch_size, shuffle=True)
    valid = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False)

    models = (
        DANNFeature(),
        DANNLabel(),
        DANNDomain()
    )
    criterions = (
        None,
        torch.nn.CrossEntropyLoss(),
        torch.nn.BCEWithLogitsLoss()
    )
    optimizers = (
        torch.optim.AdamW(models[0].parameters(), lr=args.lr),
        torch.optim.AdamW(models[1].parameters(), lr=args.lr),
        torch.optim.AdamW(models[2].parameters(), lr=args.lr)
    )
    epochs = range(args.epochs)
    if args.use_checkpoint == 'DANN':
        path = './saved_models/DANNFeature.pt'
        epoch = load_checkpoint(path, models[0], optimizers[0])
        path = './saved_models/DANNLabel.pt'
        if epoch != load_checkpoint(path, models[1], optimizers[1]):
            print('Warning: Using checkpoints from different epoch')
        path = './saved_models/DANNDomain.pt'
        if epoch != load_checkpoint(path, models[2], optimizers[2]):
            print('Warning: Using checkpoints from different epoch')
        epochs = range(epoch + 1, epoch + 1 + args.epochs)
    return (source, target, valid), models, criterions, optimizers, epochs


if __name__ == '__main__':
    main()
