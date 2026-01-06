import os 
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import numpy as np
import math
import time
from torch.cuda.amp import GradScaler, autocast

class Config:
    d_model = 1024
    nhead = 16
    num_encoder_layers = 6
    dim_feedforward = 4096
    dropout = 0.1
    batch_size = 128
    num_epochs = 100
    learning_rate = 1e-4
    weight_decay = 1e-5
    grad_clip = 0.5
    warmup_steps = 5000
    dataset_path = "data-s2s/data-train-val.json"
    model_save_path = "models/2jof_s2s.pth"
    log_path = "loss/loss_s2s.txt"
    input_token_len = 18  # h(16)+RMSD+Rg
    output_token_len = 16  # h(16)
    train_size = 160000
    val_size = 20000

config = Config()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== DATASET =====
class S2SProteinDataset(Dataset):
    def __init__(self, path, train_size, val_size):
        with open(path, 'r') as f:
            self.data = list(json.load(f).items())
        self.data = self.data[:train_size + val_size]
        self._validate_data()
        
        self.input_scaler = self._fit_minmax_scaler([self._get_input(x[1]) for x in self.data[:train_size]])
        self.output_scaler = self._fit_minmax_scaler([self._get_output(x[1]) for x in self.data[:train_size]])
        
        self.input_scaler_gpu = (
            torch.tensor(self.input_scaler[0], dtype=torch.float32, device=device),
            torch.tensor(self.input_scaler[1], dtype=torch.float32, device=device)
        )
        self.output_scaler_gpu = (
            torch.tensor(self.output_scaler[0], dtype=torch.float32, device=device),
            torch.tensor(self.output_scaler[1], dtype=torch.float32, device=device)
        )

    def _validate_data(self):
        for i, (_, rec) in enumerate(self.data[:100]):
            x = rec['Input']
            y = rec['Output']
            assert not any(np.isnan(val) for val in x['h']), f"NaN in h features at sample {i}"
            assert len(x['h']) == 16, f"Invalid h length at sample {i}"

    def _get_input(self, rec):
        x = rec['Input']
        return np.array([
            x['delta_RMSD'],
            x['delta_Rg'],
            *x['h']
        ], dtype=np.float32)

    def _get_output(self, rec):
        y = rec['Output']
        return np.array(y['h'], dtype=np.float32)

    def _fit_minmax_scaler(self, array_list):
        arr = np.stack(array_list)
        min_val = arr.min(axis=0)
        max_val = arr.max(axis=0)
        range_val = max_val - min_val
        range_val[range_val == 0] = 1.0
        return (min_val, range_val)

    def scale(self, vec, scaler):
        min_val, range_val = scaler
        return (vec - min_val) / range_val

    def inverse_scale_gpu(self, vec, scaler_gpu):
        min_val, range_val = scaler_gpu
        return vec * range_val + min_val

    def __getitem__(self, idx):
        rec = self.data[idx][1]
        x = self.scale(self._get_input(rec), self.input_scaler)
        y = self.scale(self._get_output(rec), self.output_scaler)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

# ===== Transformer-encoder =====
class ProteinTransformer(nn.Module):
    def __init__(self):
        super().__init__()

        # token embedding
        self.h_proj = nn.Sequential(
            nn.Linear(1, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
        self.condition_proj = nn.Sequential(
            nn.Linear(1, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )

        self.joint_proj = nn.Linear(256, config.d_model)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_encoder_layers
        )

        self.h_head = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, 1024),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src):
        delta_rmsd = src[:, 0:1]
        delta_rg = src[:, 1:2]
        h_feats = src[:, 2:]  #h

        # token embedding
        token_list = []
        token_list.append(self.condition_proj(delta_rmsd))  # token 1
        token_list.append(self.condition_proj(delta_rg))    # token 2
        for i in range(h_feats.shape[1]):
            token_list.append(self.h_proj(h_feats[:, i:i+1]))  # token 3~18

        tokens = torch.stack(token_list, dim=1)  # [B, 18, 256]
        x = self.joint_proj(tokens)              # [B, 18, d_model]

        encoded = self.transformer(x)            # [B, 18, d_model]
        h_tokens = encoded[:, 2:, :]
        h_out = self.h_head(h_tokens).squeeze(-1)  # [B, 16]

        return h_out

# ===== loss（only h）=====
class EnhancedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.h_loss = nn.MSELoss()
        self.h_weight = 5.0
        
    def forward(self, pred, target):
        h_loss = self.h_loss(pred, target)
        total_loss = self.h_weight * h_loss
        return total_loss, h_loss.item()

# ===== Learning rate=====
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps, max_lr):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.max_lr = max_lr
        self.current_step = 0
        
    def step(self):
        self.current_step += 1
        if self.current_step < self.warmup_steps:
            lr = self.max_lr * (self.current_step / self.warmup_steps)
        else:
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = 0.5 * self.max_lr * (1 + math.cos(math.pi * progress))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

# ===== Accuracy（Only h）=====
def compute_accuracy(pred, target, dataset):
    pred_inv = dataset.inverse_scale_gpu(pred, dataset.output_scaler_gpu)
    target_inv = dataset.inverse_scale_gpu(target, dataset.output_scaler_gpu)
    h_diff = (pred_inv - target_inv).abs()
    h_acc = (h_diff < 0.1).float().mean()
    return h_acc.item()

# ===== Train/Validation =====
def run_epoch(model, loader, optimizer, criterion, scheduler, scaler, dataset, train=True):
    model.train() if train else model.eval()
    total_loss = total_h_loss = 0
    total_h_acc = 0

    with torch.set_grad_enabled(train):
        for src, tgt in tqdm(loader, desc='Train' if train else 'Val'):
            src, tgt = src.to(device), tgt.to(device)
            if train:
                optimizer.zero_grad()
            with autocast():
                output = model(src)
                loss, h_loss = criterion(output, tgt)
            if train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
            h_acc = compute_accuracy(output, tgt, dataset)
            total_loss += loss.item()
            total_h_loss += h_loss
            total_h_acc += h_acc

    n = len(loader)
    return (total_loss/n, total_h_loss/n, total_h_acc/n)

# ===== Main function =====
if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    
    dataset = S2SProteinDataset(config.dataset_path, config.train_size, config.val_size)
    train_set, val_set = random_split(dataset, [config.train_size, config.val_size], 
                                     generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    model = ProteinTransformer().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = EnhancedLoss()
    scaler = GradScaler()
    total_steps = len(train_loader) * config.num_epochs
    scheduler = WarmupCosineScheduler(optimizer, config.warmup_steps, total_steps, config.learning_rate)
    
    os.makedirs(os.path.dirname(config.log_path), exist_ok=True)
    os.makedirs(os.path.dirname(config.model_save_path), exist_ok=True)
    
    with open(config.log_path, 'w') as f:
        f.write('Epoch,TrainLoss,TrainHLoss,TrainHAcc,ValLoss,ValHLoss,ValHAcc,Time\n')
    
    best_val_loss = float('inf')
    start_time = time.time()
    
    for epoch in range(config.num_epochs):
        epoch_start = time.time()
        train_loss, train_hl, train_ha = run_epoch(model, train_loader, optimizer, criterion, scheduler, scaler, dataset, True)
        val_loss, val_hl, val_ha = run_epoch(model, val_loader, optimizer, criterion, scheduler, scaler, dataset, False)
        epoch_time = time.time() - epoch_start
        
        with open(config.log_path, 'a') as f:
            f.write(f"{epoch+1},{train_loss:.6f},{train_hl:.6f},{train_ha:.6f},{val_loss:.6f},{val_hl:.6f},{val_ha:.6f},{epoch_time:.2f}\n")
        
        print(f"\nEpoch {epoch+1}/{config.num_epochs} | LR: {scheduler.step():.2e} | Time: {epoch_time:.2f}s")
        print(f"Train Loss: {train_loss:.4f} (H:{train_hl:.4f}) | Train H Acc: {train_ha:.2%}")
        print(f"Val   Loss: {val_loss:.4f} (H:{val_hl:.4f}) | Val   H Acc: {val_ha:.2%}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), config.model_save_path)
            print(f"Saved best model with val loss: {best_val_loss:.6f}")

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.2f} minutes")
    print(f"Best validation loss: {best_val_loss:.6f}")
