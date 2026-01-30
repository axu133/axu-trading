import torch
import torch.nn as nn
import numpy as np
from read_data import ERA5Dataset
from models import WeatherResNet3D  # Removed Weather3DCNN, EarlyStopping as not needed for inference
from timeit import default_timer
import os

# --- CONFIGURATION ---
seed_num = 1
torch.manual_seed(seed_num)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# !!! UPDATE THIS PATH TO YOUR EXISTING MODEL FILE !!!
# Example: 'weather_prediction/models/daily_max_model_1995-2025_max_ResNet_20231027.pth'
model_path_to_load = 'weather_prediction/models/daily_max_model_1980-2025_max_ResNet_20260121.pth' 


min_or_max = 'max'
years = range(1995, 2026)

if __name__ == "__main__":
    print(f"Running inference on {device}...")
    
    # 1. Prepare Data (Must match the split used during training to ensure valid testing)
    print("Loading dataset...")
    dataset = ERA5Dataset(years=years, window_size=5, max_or_min=min_or_max)
    
    total_samples = len(dataset)
    train_size = int(0.8 * total_samples)
    test_size = total_samples - train_size
    
    # We reproduce the split so we test on the actual test set, not training data
    _, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    batch_size = 48
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 2. Load Model Architecture
    model = WeatherResNet3D(input_channels=5, input_frames=20).to(device)
    
    # 3. Load Weights
    if os.path.exists(model_path_to_load):
        print(f"Loading weights from: {model_path_to_load}")
        
        # Load the dictionary
        state_dict = torch.load(model_path_to_load, map_location=device)
        
        # --- FIX START: Create a new dictionary without the 'module.' prefix ---
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                name = k[7:] # remove "module."
            else:
                name = k
            new_state_dict[name] = v
        # --- FIX END ---

        model.load_state_dict(new_state_dict)
    else:
        print(f"ERROR: Could not find model file at {model_path_to_load}")
        print("Please update the 'model_path_to_load' variable at the top of the script.")
        exit()

    loss_fn = nn.HuberLoss(delta=1.0)
    
    print("Starting evaluation...")
    model.eval()
    total_test_loss = 0.0
    
    # Storage for inspection
    all_preds = []
    all_targets = []
    all_baselines = []

    t1 = default_timer()
    
    with torch.no_grad():
        for batch_idx, (batch_data, batch_target, batch_baseline) in enumerate(test_loader):
            batch_data, batch_target, batch_baseline = (batch_data.to(device), 
                                                        batch_target.to(device).view(-1, 1), 
                                                        batch_baseline.to(device).view(-1, 1))    

            preds = model(batch_data)
            loss = loss_fn(preds, batch_target)
            total_test_loss += loss.item() * batch_data.size(0)
            
            # Store first batch for detailed inspection
            if batch_idx == 0:
                all_preds = preds.cpu().numpy().flatten()
                all_targets = batch_target.cpu().numpy().flatten()
                all_baselines = batch_baseline.cpu().numpy().flatten()

    avg_test_loss = total_test_loss / test_size
    
    print(f"\nEvaluation Complete in {default_timer()-t1:.2f}s")
    print(f"Average Test Loss: {avg_test_loss:.5f}")

    # --- INSPECTION BLOCK ---
    print("\n" + "="*60)
    print("SAMPLE PREDICTIONS (First 10 items of first batch)")
    print("="*60)
    print(f"{'Baseline':<12} | {'Target':<12} | {'Prediction':<12} | {'Diff (Pred-Tgt)':<15}")
    print("-" * 60)
    
    # Print first 10 samples
    for j in range(min(10, len(all_preds))):
        p = all_preds[j]
        t = all_targets[j]
        b = all_baselines[j]
        print(f"{b:<12.4f} | {t:<12.4f} | {p:<12.4f} | {abs(p - t):<15.4f}")

    print("-" * 60)

    # --- ZERO FUNCTION CHECK ---
    pred_std = np.std(all_preds)
    pred_mean = np.mean(all_preds)
    
    print(f"\nModel Diagnostics:")
    print(f"Prediction Mean:    {pred_mean:.4f}")
    print(f"Prediction Std Dev: {pred_std:.4f}")
    
    if pred_std < 0.01:
        print("\n⚠️  WARNING: Prediction Standard Deviation is extremely low.")
        print("    The model has likely collapsed to the mean (learning the 0 function or a constant).")
    elif pred_std < 0.5 and np.std(all_targets) > 5.0:
         print("\n⚠️  WARNING: Prediction variance is significantly lower than target variance.")
         print("    The model might be underfitting.")
    else:
        print("\n✅  PASS: The model is producing varied predictions (it is not a constant 0 function).")