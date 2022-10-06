import os

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


def mean_iou_score(prediction, target):
    prediction = prediction.cpu().numpy()
    target = target.cpu().numpy()
    mean_iou = 0
    for i in range(1, 7):
        tp_fp = np.sum(prediction == i)
        tp_fn = np.sum(target == i)
        tp = np.sum((prediction == i) * (target == i))
        iou = tp / (tp_fp + tp_fn - tp)
        mean_iou += iou / 6
    return mean_iou


def save_model(model, path):
    print(f'Saving model to {path}...')
    torch.save(model.state_dict(), path)


def train(model, train_loader, val_loader, num_epoch, device, criterion, optimizer):
    train_loss = np.zeros(num_epoch, dtype=np.float32)
    train_acc = np.zeros(num_epoch, dtype=np.float32)
    val_loss = np.zeros(num_epoch, dtype=np.float32)
    val_acc = np.zeros(num_epoch, dtype=np.float32)
    best_acc = 0

    for epoch in range(num_epoch):
        reg_loss = 0.0 
        score = 0
        model.train()
        for batch_idx, (data, label, ) in enumerate(tqdm(train_loader, postfix=f'epoch = {epoch}')):
            data = data.to(device)
            label = label.to(device)
            output = model(data)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            reg_loss += loss.item()
            score += mean_iou_score(output.argmax(dim=1), label)
        train_loss[epoch] = reg_loss / len(train_loader.dataset)
        train_acc[epoch] = score / len(train_loader.dataset)

        reg_loss = 0.0
        score = 0
        with torch.no_grad():
            model.eval()
            for batch_idx, (data, label, ) in enumerate(val_loader):
                data = data.to(device)
                label = label.to(device)
                output = model(data)
                loss = criterion(output, label)
                reg_loss += loss.item()
                reg_pred = output.argmax(dim=1)
                score += mean_iou_score(output.argmax(dim=1), label)
        val_loss[epoch] = reg_loss / len(val_loader.dataset)
        val_acc[epoch] = score / len(val_loader.dataset)

        print(f'training loss = {train_loss[epoch]:.4f}, training acc = {train_acc[epoch]:.4f}')
        print(f'validation loss = {val_loss[epoch]:.4f}, validation acc = {val_acc[epoch]:.4f}')

        if val_acc[epoch] > best_acc:
            best_acc = val_acc[epoch]
            save_model(model, './save_models/best_model.pt')
        save_model(model, f'./save_models/{epoch}.pt')
