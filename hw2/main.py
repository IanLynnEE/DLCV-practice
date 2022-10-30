import os
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from models import DCGANGenerator, DCGANDiscriminator
from models import DANNFeature, DANNLabel, DANNDomain
from datasets import FaceDataset, DigitDataset
from tools import Config
from tools import train_DCGAN, train_DDPM, train_DANN, load_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, default='face')
    parser.add_argument('--use_checkpoint', type=str, default='None')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    config = Config()
    config.use_checkpoint = args.use_checkpoint

    if 'face' in args.target:
        config.source = './hw2_data/face/train/'
        config.batch_size = 256
        config.epochs = 100
        config.lr = 0.0004
        train_DCGAN(device, *setup_DCGAN(config))

    elif 'digits' in args.target:
        config.source = './hw2_data/digits/mnistm/data/'
        config.batch_size = 256
        config.epochs = 100
        config.lr = 0.0004
        train_DDPM(device, *setup_DDPM(config))

    elif 'usps' in args.target:
        config.source = 'hw2_data/digits/mnistm/train.csv'
        config.target = 'hw2_data/digits/usps/train.csv'
        config.valid = 'hw2_data/digits/usps/val.csv'
        config.batch_size = 256
        config.epochs = 100
        config.lr = 0.001
        config.lambda_ = 0.05
        train_DANN(device, *setup_DANN(config), config.lambda_)

    elif 'svhn' in args.target:
        config.source = 'hw2_data/digits/mnistm/train.csv'
        config.target = 'hw2_data/digits/svhn/train.csv'
        config.valid = 'hw2_data/digits/svhn/val.csv'
        config.batch_size = 256
        config.epochs = 100
        config.lr = 0.001
        config.lambda_ = 0.05
        train_DANN(device, *setup_DANN(config), config.lambda_)
    return


def setup_DCGAN(config):
    # https://gist.github.com/weiaicunzai/e623931921efefd4c331622c344d8151
    means = [0.5071, 0.4865, 0.4409]
    stds = [0.2673, 0.2564, 0.2762]
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(means, stds),
        transforms.RandomHorizontalFlip(),
    ])
    train_set = FaceDataset(prefix=config.source, trans=trans)
    loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)

    models = (
        DCGANGenerator(),
        DCGANDiscriminator()
    )
    criterions = (
        torch.nn.BCELoss(),
        torch.nn.BCELoss()
    )
    optimizers = (
        torch.optim.AdamW(models[0].parameters(), lr=config.lr),
        torch.optim.AdamW(models[1].parameters(), lr=config.lr)
    )
    epochs = range(config.epochs)
    if config.use_checkpoint == 'DCGAN':
        path = './saved_models/DCGANGenerator.pt'
        epoch = load_checkpoint(path, models[0], optimizers[0])
        path = './saved_models/DCGANDiscriminator.pt'
        if epoch != load_checkpoint(path, models[1], optimizers[1]):
            print('Warning: Using checkpoints from different epoch')
        epochs = range(epoch + 1, epoch + 1 + config.epochs)
    return loader, models, criterions, optimizers, epochs


def setup_DDPM(config):
    pass


def setup_DANN(config):
    means = [0.5071, 0.4865, 0.4409]
    stds = [0.2673, 0.2564, 0.2762]
    trans = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(means, stds),
    ])
    source_set = DigitDataset(prefix=config.source, trans=trans, labeled=True)
    target_set = DigitDataset(prefix=config.target, trans=trans, labeled=False)
    valid_set = DigitDataset(prefix=config.valid, trans=trans, labeled=True)
    source = DataLoader(source_set, batch_size=config.batch_size, shuffle=True)
    target = DataLoader(target_set, batch_size=config.batch_size, shuffle=True)
    valid = DataLoader(valid_set, batch_size=config.batch_size, shuffle=False)

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
        torch.optim.AdamW(models[0].parameters(), lr=config.lr),
        torch.optim.AdamW(models[1].parameters(), lr=config.lr),
        torch.optim.AdamW(models[2].parameters(), lr=config.lr)
    )
    epochs = range(config.epochs)
    if config.use_checkpoint == 'DANN':
        path = './saved_models/DANNFeature.pt'
        epoch = load_checkpoint(path, models[0], optimizers[0])
        path = './saved_models/DANNLabel.pt'
        if epoch != load_checkpoint(path, models[1], optimizers[1]):
            print('Warning: Using checkpoints from different epoch')
        path = './saved_models/DANNDomain.pt'
        if epoch != load_checkpoint(path, models[2], optimizers[2]):
            print('Warning: Using checkpoints from different epoch')
        epochs = range(epoch + 1, epoch + 1 + config.epochs)
    return (source, target, valid), models, criterions, optimizers, epochs


if __name__ == '__main__':
    main()
