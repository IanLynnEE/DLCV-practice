import os

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


def save_model(model, path):
    print(f'Saving model to {path}...')
    torch.save(model.state_dict(), path)

def load_parameters(model, path):
    print(f'Loading model parameters from {path}...')
    param = torch.load(path)
    model.load_state_dict(param)
    # TODO: model.to(device)
    # https://www.codenong.com/cs106326580/

def train(model, train_loader, val_loader, num_epoch, device, criterion, optimizer, scheduler=None):
    train_loss = np.zeros(num_epoch, dtype=np.float32)
    train_acc = np.zeros(num_epoch, dtype=np.float32)
    val_loss = np.zeros(num_epoch, dtype=np.float32)
    val_acc = np.zeros(num_epoch, dtype=np.float32)
    best_acc = 0

    for epoch in range(num_epoch):
        reg_loss = 0.0 
        corr_num = 0
        model.train()
        for batch_idx, (data, label, ) in enumerate(tqdm(train_loader, postfix=f'epoch = {epoch}')):
            data = data.to(device)
            label = label.to(device)
            output = model(data)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            # TODO: Drop if too large?
            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.)
            optimizer.step()
            reg_loss += loss.item()
            reg_pred = output.argmax(dim=1)
            # Way to do comparison in tensor.
            corr_num += reg_pred.eq(label.view_as(reg_pred)).sum().item()
        # TODO: Adjusting learning rate?
        # scheduler.step()
        train_loss[epoch] = reg_loss / len(train_loader.dataset)
        train_acc[epoch] = corr_num / len(train_loader.dataset)

        reg_loss = 0.0
        corr_num = 0
        with torch.no_grad():
            model.eval()
            for batch_idx, (data, label, ) in enumerate(val_loader):
                data = data.to(device)
                label = label.to(device)
                output = model(data)
                loss = criterion(output, label)
                reg_loss += loss.item()
                reg_pred = output.argmax(dim=1)
                corr_num += reg_pred.eq(label.view_as(reg_pred)).sum().item()
        val_loss[epoch] = reg_loss / len(val_loader.dataset)
        val_acc[epoch] = corr_num / len(val_loader.dataset)

        print(f'training loss = {train_loss[epoch]:.4f}, training acc = {train_acc[epoch]:.4f}')
        print(f'validation loss = {val_loss[epoch]:.4f}, validation acc = {val_acc[epoch]:.4f}')

        if val_acc[epoch] > best_acc:
            best_acc = val_acc[epoch]
            save_model(model, './save_models/best_model.pt')
        save_model(model, f'./save_models/{epoch}.pt')
