import os
import argparse

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms, InterpolationMode

from myModels import myResNet50
from myDatasets import part1_dataset


def inference():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./data/p1_data/val_50')
    parser.add_argument('--model_path', type=str, default='./saved_models/ResNet50_best.pt')
    parser.add_argument('--out_path', type=str, default='./val_gt.csv')
    args = parser.parse_args()

    means = [0.5076548, 0.48128527, 0.43116662]
    stds = [0.2627228, 0.25468898, 0.27363828]
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(means, stds),
        transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
    ])

    dataset = part1_dataset(prefix=args.data_root, trans=trans, has_label=False)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

    model = myResNet50(num_out=50)
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint)

    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    model.to(device)

    preds, filenames = predict(model, dataloader, device)

    df = pd.DataFrame({
        'image_id': filenames,
        'label': preds
    })
    df = df.to_csv(args.out_path, index=False)
    return


def predict(model, dataloader, device):
    preds = []
    filenames = []

    model.eval()
    with torch.no_grad():
        for _, (data, filename) in enumerate(dataloader):
            data = data.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            preds.extend(list(pred.detach().cpu().numpy()))
            filenames.extend(list(filename))
    return preds, filenames


if __name__ == '__main__':
    inference()
