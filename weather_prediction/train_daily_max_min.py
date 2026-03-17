import torch
from torch.amp import autocast, GradScaler
import torch.nn as nn
import numpy as np
try:
    from .read_data import ERA5Dataset
    from .models import Weather3DCNN, EarlyStopping, WeatherResNet3D
except ImportError as e:  # pragma: no cover
    if "relative import" in str(e) or "no known parent package" in str(e):
        from weather_prediction.read_data import ERA5Dataset
        from weather_prediction.models import Weather3DCNN, EarlyStopping, WeatherResNet3D
    else:
        raise
from timeit import default_timer
import copy
from datetime import date
import os

seed_num = 1
torch.manual_seed(seed_num)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_directory = 'weather_prediction/models/'
date_str = date.today().strftime("%Y%m%d")
min_or_max = 'max'
years = range(1980, 2026)
comments = "_Radiation_"

if __name__ == "__main__":
    # Forecast-realistic split (by target year, not random).
    train_years = range(1980, 2018)   # 1980-2017
    val_years = range(2018, 2022)     # 2018-2021
    test_years = range(2022, 2026)    # 2022-2025

    dataset = ERA5Dataset(years=years, window_size=5, max_or_min=min_or_max, normalize=False)

    # Fit normalization ONLY on training years (no leakage), then apply to full dataset.
    train_mean, train_std = dataset.fit_normalization_for_years(train_years)
    dataset.apply_normalization(mean=train_mean, std=train_std)

    model = WeatherResNet3D(input_channels=dataset.n_channels, input_frames=20)
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        
    model = model.to(device)
    stopper = EarlyStopping(patience=55, min_delta=0.0001)  # allow more LR drops before stopping

    indices_train, indices_val, indices_test = [], [], []
    for idx in range(len(dataset)):
        _, _, _, _, y = dataset.valid_samples[idx]
        if y in train_years:
            indices_train.append(idx)
        elif y in val_years:
            indices_val.append(idx)
        elif y in test_years:
            indices_test.append(idx)

    train_dataset = torch.utils.data.Subset(dataset, indices_train)
    val_dataset = torch.utils.data.Subset(dataset, indices_val)
    test_dataset = torch.utils.data.Subset(dataset, indices_test)

    train_size = len(train_dataset)
    val_size = len(val_dataset)
    test_size = len(test_dataset)

    batch_size = 32

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory = True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory = True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory = True)

    # Climatology baseline: mean TMAX by day-of-year computed from TRAIN split only (no leakage).
    clim_sum = np.zeros(367, dtype=np.float64)
    clim_cnt = np.zeros(367, dtype=np.int64)
    for i in range(len(train_dataset)):
        _, y_abs, _, doy, _ = train_dataset[i]
        d = int(doy)
        if 1 <= d <= 366:
            clim_sum[d] += float(y_abs)
            clim_cnt[d] += 1
    clim_mean_by_doy = np.full(367, np.nan, dtype=np.float32)
    valid = clim_cnt > 0
    clim_mean_by_doy[valid] = (clim_sum[valid] / clim_cnt[valid]).astype(np.float32)

    lr = 3e-4  # lower initial LR for finer convergence
    weight_decay = 2e-4  # moderate L2 regularization
    num_epochs = 500
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=8, factor=0.5)
    loss_fn = nn.HuberLoss(delta=1.0)
    scaler = GradScaler()

    train_losses = []
    test_losses = []
    min_test_loss = np.inf

    print("Training begin")

    for i in range(num_epochs):
        model.train()
        t1 = default_timer()
        total_train_loss = 0.0
        total_train_mae_abs = 0.0
        total_train_mae_persistence = 0.0
        total_train_mae_climatology = 0.0

        for batch_data, batch_target_abs, batch_baseline_abs, batch_doy, batch_year in train_loader:
            batch_data = batch_data.to(device)
            batch_target_abs = batch_target_abs.to(device).view(-1, 1)
            batch_baseline_abs = batch_baseline_abs.to(device).view(-1, 1)
            batch_doy = batch_doy.to(device).view(-1)
            
            optimizer.zero_grad()
            
            with autocast(device_type='cuda', dtype=torch.float16):
                y = model(batch_data)
                loss = loss_fn(y, batch_target_abs)
                
            scaler.scale(loss).backward()
            
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            total_train_loss += loss.item() * batch_data.size(0)
            y_f = y.detach().float()
            total_train_mae_abs += (y_f - batch_target_abs).abs().sum().item()
            total_train_mae_persistence += (batch_baseline_abs - batch_target_abs).abs().sum().item()
            batch_clim = torch.from_numpy(clim_mean_by_doy[batch_doy.detach().cpu().numpy()]).to(device).view(-1, 1)
            total_train_mae_climatology += (batch_clim - batch_target_abs).abs().sum().item()
        
        def eval_loader(loader):
            total_loss = 0.0
            total_mae_abs = 0.0
            total_mae_persistence = 0.0
            total_mae_climatology = 0.0
            n = 0
            with torch.no_grad():
                for batch_data, batch_target_abs, batch_baseline_abs, batch_doy, batch_year in loader:
                    batch_data = batch_data.to(device)
                    batch_target_abs = batch_target_abs.to(device).view(-1, 1)
                    batch_baseline_abs = batch_baseline_abs.to(device).view(-1, 1)
                    batch_doy = batch_doy.to(device).view(-1)

                    preds = model(batch_data)
                    loss = loss_fn(preds, batch_target_abs)
                    bs = batch_data.size(0)
                    n += bs
                    total_loss += loss.item() * bs
                    preds_f = preds.detach().float()
                    total_mae_abs += (preds_f - batch_target_abs).abs().sum().item()
                    total_mae_persistence += (batch_baseline_abs - batch_target_abs).abs().sum().item()
                    batch_clim = torch.from_numpy(clim_mean_by_doy[batch_doy.detach().cpu().numpy()]).to(device).view(-1, 1)
                    total_mae_climatology += (batch_clim - batch_target_abs).abs().sum().item()
            return (
                total_loss / max(n, 1),
                total_mae_abs / max(n, 1),
                total_mae_persistence / max(n, 1),
                total_mae_climatology / max(n, 1),
            )

        model.eval()
        avg_val_loss, avg_val_mae_abs, avg_val_mae_persist, avg_val_mae_climo = eval_loader(val_loader)
        avg_test_loss, avg_test_mae_abs, avg_test_mae_persist, avg_test_mae_climo = eval_loader(test_loader)
            
        avg_train_loss = total_train_loss / train_size
        avg_train_mae_abs = total_train_mae_abs / train_size
        avg_train_mae_persistence = total_train_mae_persistence / train_size
        avg_train_mae_climatology = total_train_mae_climatology / train_size
        avg_test_mae_abs = avg_test_mae_abs
        avg_test_mae_persistence = avg_test_mae_persist
        avg_test_mae_climatology = avg_test_mae_climo

        train_losses.append(avg_train_loss)
        test_losses.append(avg_val_loss)

        if avg_val_loss < min_test_loss:
            min_test_loss = avg_val_loss
            best_model_state = copy.deepcopy(model)
        
        scheduler.step(avg_val_loss)
        lr = scheduler.get_last_lr()[0]
        elapsed = default_timer() - t1
        print(
            f"Epoch: {i + 1:3d}, "
            f"Train Loss: {avg_train_loss:8.4f}, Val Loss: {avg_val_loss:8.4f}, "
            f"Train MAE_abs: {avg_train_mae_abs:6.3f}, Val MAE_abs: {avg_val_mae_abs:6.3f}, "
            f"Val MAE_persist: {avg_val_mae_persist:6.3f}, Val MAE_climo: {avg_val_mae_climo:6.3f}, "
            f"Test MAE_abs: {avg_test_mae_abs:6.3f}, Test MAE_persist: {avg_test_mae_persistence:6.3f}, Test MAE_climo: {avg_test_mae_climatology:6.3f}, "
            f"LR: {lr:10.2e}, Time: {elapsed:6.1f}s"
        )

        if stopper(avg_test_loss):
            print("Early stopping triggered.")
            break

    model_path = os.path.join(model_directory, f'daily_max_model_{min(years)}-{max(years)}_{min_or_max}{comments}{date_str}.pth')
    torch.save(best_model_state.state_dict(), model_path)
            
