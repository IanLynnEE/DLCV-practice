import os
import argparse

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.utils import save_image

from datasets import DigitDataset
from models import DCGANGenerator
from models import UNet
from models import DANNFeature, DANNLabel


def inference():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='face')
    parser.add_argument('--input_dir', type=str, default='None')
    parser.add_argument('--output_dir_or_path', type=str)
    parser.add_argument('--seed', type=int, default=54)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

    if args.output_dir_or_path.endswith('.csv'):
        pass
    elif not os.path.isdir(args.output_dir_or_path):
        os.mkdir(args.output_dir_or_path)

    # Part 3
    if args.input_dir != 'None':
        if 'usps' in args.input_dir:
            model_dir = 'saved_models/DANN_usps/'
        elif 'svhn' in args.input_dir:
            model_dir = 'saved_models/DANN_svhn/'
        else:
            raise NotImplementedError
        filenames, preds = DANN_predict(*setup_DANN(device, model_dir, args.input_dir))
        df = pd.DataFrame({
            'image_name': filenames,
            'label': preds
        })
        df = df.to_csv(args.output_dir_or_path, index=False)

    # Part 1
    elif args.data == 'face':
        model_path = 'saved_models/DCGAN/DCGANGenerator_149.pt'
        model = DCGANGenerator()
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device).eval()
        with torch.no_grad():
            if torch.cuda.is_available:
                torch.cuda.manual_seed(args.seed)
            noise = torch.randn((1000, 100, 1, 1), device=device)
            output = model(noise)
        for i, image in enumerate(output):
            save_image((image + 1) / 2, os.path.join(args.output_dir_or_path, f'{i:03d}.png'))

    # Part 2
    elif args.data == 'digits':
        model_path = 'saved_models/DDPM_EMA/UNet_99.pt'
        noise_steps = 1000
        beta_start = 0.0001
        beta_end = 0.02
        post_trans = transforms.Compose([
            transforms.CenterCrop(28),
            transforms.Normalize((-1, -1, -1), (2, 2, 2)),
        ])
        model = UNet(c_in=3, c_out=3, time_dim=256, num_classes=10)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device).eval()
        beta = torch.linspace(beta_start, beta_end, noise_steps)
        beta = beta.to(device)
        alpha = 1 - beta
        alpha_hat = torch.cumprod(alpha, dim=0)

        ones = torch.ones(100, dtype=torch.long, device=device)
        with torch.no_grad():
            for j in range(10):
                labels = j * ones
                x = torch.randn((100, 3, 32, 32), device=device)
                for i in tqdm(reversed(range(0, noise_steps))):
                    t = i * ones
                    predicted_noise = model(x, t, labels)
                    beta_t = beta[t][:, None, None, None]
                    alpha_t = alpha[t][:, None, None, None]
                    alpha_hat_t = alpha_hat[t][:, None, None, None]
                    noise = torch.sqrt(beta_t) * torch.randn_like(x) if i > 1 else torch.zeros_like(x)
                    x = 1 / torch.sqrt(alpha_t) * (x - beta_t / torch.sqrt(1 - alpha_hat_t) * predicted_noise) + noise
                    # if i % 200 == 0 or i == 999:
                    #     save_image(post_trans(x[0]), f'figures/noise_{j}_{i}.png')
                for i, image in enumerate(x):
                    save_image(post_trans(image), os.path.join(args.output_dir_or_path, f'{j}_{i:03d}.png'))
    else:
        raise NotImplementedError
    return


def setup_DANN(device, model_dir, data_dir):
    mean = (0.4632221, 0.4668803, 0.41948238)
    std = (0.2537196, 0.23820335, 0.2622173)
    trans = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    # Set to be true for visualize only
    target_set = DigitDataset(prefix=data_dir, trans=trans, labeled=False)
    target = DataLoader(target_set, batch_size=16, shuffle=False)

    model0 = DANNFeature()
    checkpoint = torch.load(os.path.join(model_dir, 'DANNFeature.pt'))
    model0.load_state_dict(checkpoint['model_state_dict'])
    model1 = DANNLabel()
    checkpoint = torch.load(os.path.join(model_dir, 'DANNLabel.pt'))
    model1.load_state_dict(checkpoint['model_state_dict'])

    # source_set = DigitDataset(prefix='hw2_data/digits/mnistm/train.csv', trans=trans, labeled=True)
    # source = DataLoader(source_set, batch_size=16, shuffle=False)
    # visualize(model0, source, target, device)

    return device, model0, model1, target


def DANN_predict(device, model0, model1, loader):
    preds = []
    filenames = []

    model0.to(device).eval()
    model1.to(device).eval()
    with torch.no_grad():
        for (data, filename) in loader:
            data = data.to(device)
            output = model1(model0(data))
            pred = output.argmax(dim=1)
            preds.extend(list(pred.detach().cpu().numpy()))
            filenames.extend(list(filename))
    return filenames, preds


def plot_tsne(hiddens, labels, classes=10):
    # Ref: https://blog.csdn.net/hustqb/article/details/80628721
    tsne = TSNE(2)
    components = tsne.fit_transform(hiddens)

    df = pd.DataFrame(data=components, columns=['component 1', 'component 2'])
    df['target'] = labels

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Component 1', fontsize=15)
    ax.set_ylabel('Component 2', fontsize=15)
    ax.set_title('tSNE', fontsize=20)
    for target in range(classes):
        indices = df['target'] == target
        ax.scatter(df.loc[indices, 'component 1'], df.loc[indices, 'component 2'])
    plt.show()
    return


def visualize(model, source, target, device):
    hiddens = []
    labels = []
    model.to(device).eval()
    with torch.no_grad():
        for (data, label) in target:
            data = data.to(device)
            output = model(data)
            hiddens.append(output.detach().cpu())
            labels.extend(list(label))
        features = torch.cat(hiddens)
        plot_tsne(features, labels)

        for (data, label) in source:
            data = data.to(device)
            output = model(data)
            hiddens.append(output.detach().cpu())
            labels.extend(list(label))
        hiddens = torch.cat(hiddens)
    zeros = [0 for _ in range(len(target.dataset))]
    ones = [1 for _ in range(len(source.dataset))]
    domain = zeros + ones
    plot_tsne(hiddens, domain, 2)
    return


if __name__ == '__main__':
    inference()
