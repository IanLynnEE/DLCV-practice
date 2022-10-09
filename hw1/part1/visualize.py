import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from myModels import myLeNet
from myDatasets import part1_dataset


def visualize():
    model = myLeNet(num_out=50)
    checkpoint = torch.load('./saved_models/LeNet.pt')

    means = [0.5076548, 0.48128527, 0.43116662]
    stds = [0.2627228, 0.25468898, 0.27363828]
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(means, stds),
    ])

    dataset = part1_dataset(prefix='./data/p1_data/val_50/', trans=trans, has_label=True)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

    model.load_state_dict(checkpoint)
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    model.to(device)

    hiddens, labels = predict(model, dataloader, device)
    plot_PCA(hiddens, labels)
    plot_tsne(hiddens, labels)


def plot_PCA(hiddens, labels):
    # Ref: https://towardsdatascience.com/e653f8989e60
    pca = PCA(n_components=2)
    components = pca.fit_transform(hiddens)

    df = pd.DataFrame(data=components, columns=['component 1', 'component 2'])
    df['target'] = labels

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('PCA', fontsize=20)
    for target in range(50):
        indices = df['target'] == target
        ax.scatter(df.loc[indices, 'component 1'], df.loc[indices, 'component 2'])
    plt.show()
    return


def plot_tsne(hiddens, labels):
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
    for target in range(50):
        indices = df['target'] == target
        ax.scatter(df.loc[indices, 'component 1'], df.loc[indices, 'component 2'])
    plt.show()
    return


def predict(model, dataloader, device):
    activation = {}
    hiddens = []
    labels = []

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    model.eval()
    model.fc2.register_forward_hook(get_activation('linear'))

    with torch.no_grad():
        for (data, label) in dataloader:
            data = data.to(device)
            model(data)
            hiddens.append(activation['linear'].detach().cpu())
            labels.extend(list(label))
        hiddens = torch.cat(hiddens)
    return hiddens, labels


if __name__ == '__main__':
    visualize()
