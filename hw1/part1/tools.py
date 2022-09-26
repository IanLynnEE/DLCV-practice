import os
import time

import numpy as np
import torch
import torch.nn as nn


def save_model(model, path):
    print(f'Saving model to {path}...')
    torch.save(model.state_dict(), path)
    print('End of saving.')

def load_parameters(model, path):
    print(f'Loading model parameters from {path}...')
    # TODO: how many gpu to use?
    param = torch.load(path, map_location={'cuda:0': 'cuda:1'})
    model.load_state_dict(param)
    print('End of loading.')

def train(model, train_loader, val_loader, num_epoch, device, criterion, optimizer, scheduler=None):
    train_loss = np.zeros(num_epoch, dtype=np.float32)
    train_acc = np.zeros(num_epoch, dtype=np.float32)
    val_loss = np.zeros(num_epoch, dtype=np.float32)
    val_acc = np.zeros(num_epoch, dtype=np.float32)
    best_acc = 0
    start_train = time.time()

    for epoch in range(num_epoch):
        print(f'epoch = {epoch}')
        start_time = time.time()
        reg_loss = 0.0 
        corr_num = 0
        model.train()
        for batch_idx, (data, label, ) in enumerate(train_loader):
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
                data.to(device)
                label.to(device)
                output = model(data)
                loss = criterion(output, label)
                reg_loss += loss.item()
                reg_pred = output.argmax(dim=1)
                corr_num += reg_pred.eq(label.view_as(reg_pred)).sum().item()
        val_loss[epoch] = reg_loss / len(val_loader.dataset)
        val_acc[epoch] = corr_num / len(val_loader.dataset)

        end_time = time.time()
        print(f'\r, time = {(time.time() - start_time) // 60} MIN')
        print(f'training loss = {train_loss[epoch]:.4f}, training acc = {train_acc[epoch]:.4f}')
        print(f'validation loss = {val_loss[epoch]:.4f}, validation acc = {val_acc[epoch]:.4f}')

        if val_acc > best_acc:
            best_acc = val_acc
            save_model(model, './models/best_model.pt')
        save_model(model, f'./models/{epoch}.pt')