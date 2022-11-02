import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from tqdm import tqdm
from pytorch_fid.fid_score import calculate_fid_given_paths

from face_recog import face_recog
from digit_classifier import classify_dir


class Config:
    def __init__(self):
        self.source = None
        self.target = None
        self.valid = None
        self.epochs = 1
        self.batch_size = 16
        self.lr = 0.001
        self.betas = (0.9, 0.999)
        self.lambda_ = 0.05
        self.beta_start = 1e-4
        self.beta_end = 2e-2
        self.noise_steps = 1000
        self.use_checkpoint = False


def fix_seed(seed):
    pass


def save_checkpoint(epoch, model, optimizer, scheduler=None):
    path = f'saved_models/{model.__class__.__name__}_{epoch}.pt'
    print(f'Saving model to {path}...')
    if scheduler is None:
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            },
            path
        )
    else:
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()
            },
            path
        )


def load_checkpoint(path, model, optimizer, scheduler=None):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is None:
        print(model.state_dict, optimizer.state_dict)
        return checkpoint['epoch']
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    print(model.state_dict, optimizer.state_dict, scheduler.state_dict)
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


def train_DCGAN(device, loader, models, criterions, optimizers, epochs, post_trans):
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
            ones = torch.ones((batch_size, 1, 1, 1), dtype=torch.float, device=device)
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
            loss_g = criterions[0](output, ones)
            sum_loss_g += loss_g.item()
            loss_g.backward()
            optimizer_g.step()

        generator.eval()
        with torch.no_grad():
            noise = torch.randn((1000, 100, 1, 1), device=device)
            output = generator(noise)
            for i, image in enumerate(output):
                save_image(post_trans(image), f'outputs/hw2_1/{i:04d}.png')

        fid = calculate_fid_given_paths(
            ('outputs/hw2_1/', 'hw2_data/face/val/'),
            50,
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

        if (epoch % 5 == 4) or (best_score < hog):
            save_checkpoint(epoch, generator, optimizer_g)
            save_checkpoint(epoch, discriminator, optimizer_d)
            best_score = hog if best_score < hog else best_score
    writer.close()


def train_DANN(device, loaders, models, criterions, optimizers, epochs, lambda_):
    writer = SummaryWriter('saved_models/')
    best_score = 0.0

    for model in models:
        model.to(device)

    for optimizer in optimizers:
        optimizer_to(optimizer, device)

    target_iter = iter(loaders[1])
    for epoch in epochs:
        sum_loss_class = 0.0
        sum_loss_domain = 0.0
        sum_loss = 0.0
        sum_loss_class_valid = 0.0
        num_correct = 0

        for model in models:
            model.train()
        for source, label_class in tqdm(loaders[0], postfix=f'epoch = {epoch}'):
            source = source.to(device)
            label_class = label_class.to(device)

            # Get mini-batch of target images
            try:
                target = next(target_iter)
            except StopIteration:
                target_iter = iter(loaders[1])
                target = next(target_iter)
            target = target[0].to(device)

            # Soft domain labels
            label_positive = 1.0 - 0.3 * torch.rand((source.size(0), 1), dtype=torch.float, device=device)
            label_negative = 0.0 + 0.3 * torch.rand((target.size(0), 1), dtype=torch.float, device=device)

            # Cat source and target
            data = torch.cat([source, target], dim=0)
            label_domain = torch.cat([label_positive, label_negative], dim=0)

            # Get feature
            feature = models[0](data)
            feature_source = feature[:source.size(0)]

            # Train domain classifier
            optimizers[2].zero_grad()
            output = models[2](feature)
            loss_domain = criterions[2](output, label_domain)
            loss_domain.backward(retain_graph=True)
            sum_loss_domain += loss_domain.item()
            optimizers[2].step()

            # Train feature extractor and class classifier
            optimizers[0].zero_grad()
            optimizers[1].zero_grad()
            output = models[1](feature_source)
            loss_class = criterions[1](output, label_class)
            output = models[2](feature)
            loss_domain = criterions[2](output, label_domain)
            loss = loss_class - lambda_ * loss_domain
            loss.backward()
            sum_loss_class += loss_class.item()
            sum_loss += loss.item()
            optimizers[0].step()
            optimizers[1].step()

        for model in models:
            model.eval()
        with torch.no_grad():
            for data, label in loaders[2]:
                data = data.to(device)
                label = label.to(device)
                output = models[1](models[0](data))
                loss = criterions[1](output, label)
                sum_loss_class_valid += loss.item()
                pred = output.argmax(dim=1)
                num_correct += pred.eq(label.view_as(pred)).sum().item()

        writer.add_scalar('loss/loss_domain', sum_loss_domain, epoch)
        writer.add_scalar('loss/loss_class', sum_loss_class, epoch)
        writer.add_scalar('loss/loss_total', sum_loss, epoch)
        writer.add_scalar('loss/loss_valid', sum_loss_class_valid, epoch)
        writer.add_scalar('accuracy', num_correct / len(loaders[2].dataset), epoch)
        print(f'epoch = {epoch}, accuracy = {num_correct / len(loaders[2].dataset)}')

        if (epoch % 10 == 9) or (best_score < num_correct):
            save_checkpoint(epoch, models[0], optimizers[0])
            save_checkpoint(epoch, models[1], optimizers[1])
            save_checkpoint(epoch, models[2], optimizers[2])
            best_score = num_correct if best_score < num_correct else best_score
    writer.close()


# https://github.com/tcapelle/Diffusion-Models-pytorch
def train_DDPM(device, loader, model, criterion, optimizer, scheduler, beta, noise_steps, epochs, post_trans):
    writer = SummaryWriter('saved_models/')

    model.to(device)
    beta = beta.to(device)
    alpha = 1 - beta
    alpha_hat = torch.cumprod(alpha, dim=0)

    for epoch in epochs:
        sum_loss = 0.0
        model.train()
        for images, labels in tqdm(loader, postfix=f'epoch = {epoch}'):
            images = images.to(device)
            labels = labels.to(device)
            batch_size = images.size(0)

            t = torch.randint(1, noise_steps, (batch_size,), device=device)
            x_t, noise = noise_images(images, t, alpha_hat)

            optimizer.zero_grad()
            # Classifier Free Guidance
            if torch.rand(1) < 0.1:
                labels = None
            predicted_noise = model(x_t, t, labels)
            loss = criterion(noise, predicted_noise)
            sum_loss += loss
            loss.backward()
            optimizer.step()
            scheduler.step()

        writer.add_scalar('lr', scheduler.get_last_lr()[0], epoch)
        writer.add_scalar('loss', sum_loss.mean().item(), epoch)

        if epoch % 5 != 4:
            continue

        # 8 images per class
        model.eval()
        with torch.no_grad():
            for j in range(10):
                labels = j * torch.ones(8, dtype=torch.long, device=device)
                x = torch.randn((8, 3, 32, 32), device=device)
                for i in tqdm(reversed(range(1, noise_steps)), postfix=f'For class {j}'):
                    t = i * torch.ones(8, dtype=torch.long, device=device)
                    predicted_noise = model(x, t, labels)

                    # Classifier Free Guidance
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, 3)

                    beta_t = beta[t][:, None, None, None]
                    alpha_t = alpha[t][:, None, None, None]
                    alpha_hat_t = alpha_hat[t][:, None, None, None]
                    noise = torch.sqrt(beta_t) * torch.randn_like(x) if i > 1 else torch.zeros_like(x)
                    x = 1 / torch.sqrt(alpha_t) * (x - beta_t / torch.sqrt(1 - alpha_hat_t) * predicted_noise) + noise
                for i, img in enumerate(x):
                    save_image(post_trans(img), f'outputs/hw2_2/{j}_{i+1:03d}.png')
        score = classify_dir('outputs/hw2_2/')
        writer.add_scalar('score', score, epoch)
        save_checkpoint(epoch, model, optimizer, scheduler)
    writer.close()


def noise_images(x, t, alpha_hat):
    sqrt_alpha_hat = torch.sqrt(alpha_hat[t])[:, None, None, None]
    sqrt_one_minus_alpha_hat = torch.sqrt(1 - alpha_hat[t])[:, None, None, None]
    E = torch.randn_like(x)
    return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * E, E
