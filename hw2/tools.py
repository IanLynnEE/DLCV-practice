import subprocess

import torch
from torchvision.utils import save_image 
from tqdm import tqdm
from pytorch_fid.fid_score import calculate_fid_given_paths


def fix_seed(seed):
    pass


def save_model(model, path):
    print(f'Saving model to {path}...')
    # TODO: Need to use better method.
    torch.save(model.state_dict(), path)


def train_GAN(device, loader, models, criterions, optimizers, epochs, batch_size):
    # TODO: Tensorboard?
    best_score = 0.0

    generator = models[0]
    discriminator = models[1]
    generator.to(device)
    discriminator.to(device)

    optimizer_g = optimizers[0]
    optimizer_d = optimizers[1]

    for epoch in range(epochs):
        generator.train()
        discriminator.train()

        for data in tqdm(loader, postfix=f'epoch = {epoch}'):
            data = data.to(device)

            # TODO: Add noise to the discriminator?
            label_real = 1.0 - 0.3 * torch.rand((batch_size,), dtype=torch.float, device=device)
            label_fake = 0.0 + 0.3 * torch.rand((batch_size,), dtype=torch.float, device=device)
            noise = torch.randn((batch_size, 100, 1, 1), device=device)
            
            # Train discriminator
            optimizer_d.zero_grad()
            output = discriminator(data)
            loss_d_real = criterions[1](output, label_real)

            image_fake = generator(noise)
            output = discriminator(image_fake)
            loss_d_fake = criterions[1](output, label_fake)

            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            optimizer_d.step()

            # Train generator
            optimizer_g.zero_grad()
            output = discriminator(image_fake)
            loss_g = criterions[0](output, label_real)
            loss_g.backward()
            optimizer_g.step()

        generator.eval()
        with torch.no_grad():
            # TODO: Reduce batch size to speed up.
            noise = torch.randn((1000, 100, 1, 1), device=device)
            output = generator(noise)
            for i, image in enumerate(output):
                save_image((image + 1) / 2, f'outputs/hw2_1/{i:4d}.png')

        # TODO: Two subprocesses to calculate score
        fid = calculate_fid_given_paths(
                ('outputs/hw2_1/','hw2_data/face/test/'),
                32,
                device,
                2048,
                4
            )
        subprocess.run(['python', 'face_recog.py', '--image_dir', 'outputs/hw2_1/'])