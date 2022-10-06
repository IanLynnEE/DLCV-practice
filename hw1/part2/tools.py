import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def save_model(model, path):
    print(f'Saving model to {path}...')
    torch.save(model.state_dict(), path)


def train(model, train_loader, val_loader, num_epoch, device, criterion, optimizer):
    train_loss = np.zeros(num_epoch, dtype=np.float32)
    train_score = np.zeros(num_epoch, dtype=np.float32)
    val_loss = np.zeros(num_epoch, dtype=np.float32)
    val_score = np.zeros(num_epoch, dtype=np.float32)
    best_acc = 0

    for epoch in range(num_epoch):
        reg_loss = 0.0 
        tracker = MetricTracker()
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
            tracker.update(label, output.argmax(dim=1))
        train_loss[epoch] = reg_loss / len(train_loader.dataset)
        train_score[epoch] = tracker.get_result()

        reg_loss = 0.0
        tracker.reset()
        with torch.no_grad():
            model.eval()
            for batch_idx, (data, label, ) in enumerate(val_loader):
                data = data.to(device)
                label = label.to(device)
                output = model(data)
                loss = criterion(output, label)
                reg_loss += loss.item()
                tracker.update(label, output.argmax(dim=1))
        val_loss[epoch] = reg_loss / len(val_loader.dataset)
        val_score[epoch] = tracker.get_result()

        print(f'training loss = {train_loss[epoch]:.4f}, training acc = {train_score[epoch]:.4f}')
        print(f'validation loss = {val_loss[epoch]:.4f}, validation acc = {val_score[epoch]:.4f}')

        if val_score[epoch] > best_acc:
            best_acc = val_score[epoch]
            save_model(model, './save_models/best_model.pt')
        save_model(model, f'./save_models/{epoch}.pt')


class MetricTracker:
    def __init__(self):
        self._data = pd.DataFrame(index=[0, 1, 2, 3, 4, 5], columns=['overlap', 'union'])
        self.reset()
        
    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, target, prediction):
        target = target.cpu().numpy()
        prediction = prediction.cpu().numpy()
        for i in range(6):
            self._data['overlap'][i] += np.logical_and(target == i, prediction == i)
            self._data['union'][i] += np.logical_or(target == i, prediction == i)
    
    def get_result(self):
        self._data['iou'] = self._data['overlap'] / self._data['union']
        return self._data['iou'].mean()
