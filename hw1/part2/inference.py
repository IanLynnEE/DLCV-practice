import os
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.utils import save_image

from my_models import VGG16_FCN8s, VGG16_FCN32s
from my_datasets import part2_dataset


def inference():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./data/p2_data/validation')
    parser.add_argument('--model_path', type=str, default='./saved_models/VGG16_FCN8s.pt')
    parser.add_argument('--out_root', type=str, default='./data/p2_data/prediction')
    args = parser.parse_args()

    if not os.path.isdir(args.out_root):
        os.mkdir(args.out_root)

    means = [0.40851322, 0.37851325, 0.28088534]
    stds = [0.14234462, 0.10848372, 0.09824718]
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(means, stds),
    ])
    dataset = part2_dataset(prefix=args.data_root, trans=trans, has_mask=False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = VGG16_FCN8s(num_classes=7)
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint)

    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    model.to(device)

    preds, filenames = predict(model, dataloader, device)
    rgb_masks = label2rgb(preds)
    for i, pred in enumerate(rgb_masks):
        filename = filenames[i].replace('_sat.jpg', '.png')
        save_image(pred, os.path.join(args.out_root, filename))
    return


def predict(model, dataloader, device):
    outputs = []
    filenames = []

    model.eval()
    with torch.no_grad():
        for (data, filename) in dataloader:
            data = data.to(device)
            output = model(data)
            outputs.append(output.detach().cpu())
            filenames.extend(list(filename))
        preds = torch.cat(outputs)
        preds = preds.argmax(dim=1)
    return preds, filenames


def label2rgb(labels):
    rgb = torch.zeros(labels.size(0), 3, labels.size(1), labels.size(2))

    remap_masks = labels.to(torch.int8)
    remap_masks[labels == 0] = 3    # (Cyan: 011) Urban land
    remap_masks[labels == 1] = 6    # (Yellow: 110) Agriculture land
    remap_masks[labels == 2] = 5    # (Purple: 101) Rangeland
    remap_masks[labels == 3] = 2    # (Green: 010) Forest land
    remap_masks[labels == 4] = 1    # (Blue: 001) Water
    remap_masks[labels == 5] = 7    # (White: 111) Barren land
    remap_masks[labels == 6] = 0    # (Black: 000) Unknown

    rgb[:, 0, :, :] = remap_masks // 4
    remap_masks = remap_masks % 4
    rgb[:, 1, :, :] = remap_masks // 2
    remap_masks = remap_masks % 2
    rgb[:, 2, :, :] = remap_masks

    return rgb * 255


if __name__ == '__main__':
    inference()
