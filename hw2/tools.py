import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from tqdm import tqdm
from pytorch_fid.fid_score import calculate_fid_given_paths

from face_recog import face_recog


def fix_seed(seed):
    pass


def save_checkpoint(epoch, model, optimizer):
    path = f'saved_models/{model.__class__.__name__}_{epoch}.pt'
    print(f'Saving model to {path}...')
    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        },
        path
    )


def load_checkpoint(path, model, optimizer):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch']


# https://github.com/pytorch/pytorch/issues/8741
def optimizer_to(optim, device):
    for param in optim.state.values():
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)
    return


def train_GAN(device, loader, models, criterions, optimizers, epochs):
    writer = SummaryWriter('saved_models/')
    best_score = 0.0

    generator = models[0]
    discriminator = models[1]
    generator.to(device)
    discriminator.to(device)

    optimizer_g = optimizers[0]
    optimizer_d = optimizers[1]
    optimizer_to(optimizer_g, device)
    optimizer_to(optimizer_d, device)

    for epoch in epochs:
        sum_loss_d_real = 0.0
        sum_loss_d_fake = 0.0
        sum_loss_g = 0.0

        generator.train()
        discriminator.train()
        for data in tqdm(loader, postfix=f'epoch = {epoch}'):
            data = data.to(device)
            batch_size = data.size(0)

            positive = 1.0 - 0.3 * torch.rand((batch_size, 1, 1, 1), dtype=torch.float, device=device)
            negative = 0.0 + 0.3 * torch.rand((batch_size, 1, 1, 1), dtype=torch.float, device=device)
            noise = torch.randn((batch_size, 100, 1, 1), device=device)

            # Train discriminator
            optimizer_d.zero_grad()
            output = discriminator(data)
            loss_d_real = criterions[1](output, positive)
            sum_loss_d_real += loss_d_real.item()

            image_fake = generator(noise)
            output = discriminator(image_fake)
            loss_d_fake = criterions[1](output, negative)
            sum_loss_d_fake += loss_d_fake.item()

            loss_d = loss_d_real + loss_d_fake
            loss_d.backward(retain_graph=True)
            optimizer_d.step()

            # Train generator
            optimizer_g.zero_grad()
            output = discriminator(image_fake)
            loss_g = criterions[0](output, positive)
            sum_loss_g += loss_g.item()
            loss_g.backward()
            optimizer_g.step()

        generator.eval()
        with torch.no_grad():
            noise = torch.randn((1000, 100, 1, 1), device=device)
            output = generator(noise)
            for i, image in enumerate(output):
                save_image(image, f'outputs/hw2_1/{i:04d}.png')

        fid = calculate_fid_given_paths(
            ('outputs/hw2_1/', 'hw2_data/face/val/'),
            32,
            device,
            2048
        )
        hog = face_recog('outputs/hw2_1/')
        print(f'epoch = {epoch:3d}, fid = {fid:3.2f}, hog = {hog:3.2f}')

        writer.add_scalar('loss_d/loss_d_real', sum_loss_d_real, epoch)
        writer.add_scalar('loss_d/loss_d_fake', sum_loss_d_fake, epoch)
        writer.add_scalar('loss_g/loss_g', sum_loss_g, epoch)
        writer.add_scalar('FID', fid, epoch)
        writer.add_scalar('HOG', hog, epoch)

        if (epoch % 5 == 0) or (best_score < fid):
            save_checkpoint(epoch, generator, optimizer_g)
            save_checkpoint(epoch, discriminator, optimizer_d)
            best_score = fid if best_score < fid else None
    writer.close()
