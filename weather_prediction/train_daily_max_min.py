import torch
import torch.nn as nn
import numpy as np
from read_data import ERA5Dataset
from models import Weather3DCNN, EarlyStopping
from timeit import default_timer
import copy
from datetime import date
import os

seed_num = 1
torch.manual_seed(seed_num)
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
model_directory = 'weather_prediction/models/'
date_str = date.today().strftime("%Y%m%d")
min_or_max = 'max'
years = range(2015, 2026)

if __name__ == "__main__":
    dataset = ERA5Dataset(years=years, window_size=5, max_or_min=min_or_max)

    model = Weather3DCNN(input_channels=5, input_frames=20).to(device)
    stopper = EarlyStopping(patience=10, min_delta=0.0001)

    total_samples = len(dataset)
    train_size = int(0.8 * total_samples)
    test_size = total_samples - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    batch_size = 64

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    lr = 3e-4
    weight_decay = 1e-4
    num_epochs = 500
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=8, factor=0.5)
    loss_fn = nn.HuberLoss(delta=1.0)

    train_losses = []
    test_losses = []
    min_test_loss = np.inf

    print("Training begin")

    for i in range(num_epochs):
        model.train()
        t1 = default_timer()
        total_train_loss = 0.0
        for batch_data, batch_target in train_loader:
            batch_data, batch_target = batch_data.to(device), batch_target.to(device)
            
            y = model(batch_data)
            loss = loss_fn(y, batch_target.view(-1, 1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item() * batch_data.size(0)
        
        model.eval()
        total_test_loss = 0.0
        with torch.no_grad():
            for batch_data, batch_target in test_loader:
                batch_data, batch_target = batch_data.to(device), batch_target.to(device)

                preds = model(batch_data)
                loss = loss_fn(preds, batch_target.view(-1, 1))
                total_test_loss += loss.item() * batch_data.size(0)
            
        avg_train_loss = total_train_loss / train_size
        avg_test_loss = total_test_loss / test_size

        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)

        if avg_test_loss < min_test_loss:
            min_test_loss = avg_test_loss
            best_model_state = copy.deepcopy(model)
        
        scheduler.step(avg_test_loss)
        print(f"Epoch: {i + 1}, Avg Train Loss: {avg_train_loss}, Avg Test Loss: {avg_test_loss}, LR: {scheduler.get_last_lr()}, Time taken: {default_timer()-t1}")

        if stopper(avg_test_loss):
            print("Early stopping triggered.")
            break

    model_path = os.path.join(model_directory, f'daily_max_model_{min(years)}-{max(years)}_{min_or_max}_{date_str}.pth')
    torch.save(best_model_state.state_dict(), model_path)
            
