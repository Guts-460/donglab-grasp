import os
import json
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast

# =========================================================
# Configurations
# =========================================================
class Config:
    d_model = 1024
    nhead = 16 
    num_encoder_layers = 6
    dim_feedforward = 4096
    dropout = 0.1
    d_proj = 256
    d_h_head = 512
    batch_size = 128
    num_epochs = 100
    learning_rate = 1e-4
    weight_decay = 1e-5
    grad_clip = 0.5
    warmup_steps = 5000

    dataset_path = "data-mapping/pre_pairwise.json"
    model_save_path = "models/mapping-10-1.pth"
    log_path = "loss/loss_mapping-10-1.txt"

    input_token_len = 18   # 2 + 16
    output_token_len = 16

    train_size = 8000
    val_size = 2000


config = Config()
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# =========================================================
# Dataset
# =========================================================
class MappingProteinDataset(Dataset):
    def __init__(self, path, train_size, val_size):
        with open(path, "r") as f:
            raw = json.load(f)

        records = list(raw.values())
        assert len(records) > 1, "pre_pairwise.json hasn't sufficient pairwise data"

        self.N = train_size + val_size

        rmsd = np.array([r["data"]["RMSD"] for r in records], dtype=np.float32)
        rg   = np.array([r["data"]["Rg"]   for r in records], dtype=np.float32)
        h    = np.array([r["data"]["Mu"] + r["data"]["Logvar"] for r in records],dtype=np.float32)

        idx_i = np.random.randint(0, len(records), size=self.N)
        idx_j = np.random.randint(0, len(records), size=self.N)

        self.inputs = np.zeros((self.N, config.input_token_len), dtype=np.float32)
        self.outputs = np.zeros((self.N, config.output_token_len), dtype=np.float32)

        self.inputs[:, 0] = rmsd[idx_j] - rmsd[idx_i] #!! ΔCVij
        self.inputs[:, 1] = rg[idx_j]   - rg[idx_i] #!! ΔCVij
        self.inputs[:, 2:] = h[idx_i]
        self.outputs[:] = h[idx_j]

        # ===== scaler（仅训练集）=====
        self.input_scaler = self._fit_minmax(self.inputs[:train_size])
        self.output_scaler = self._fit_minmax(self.outputs[:train_size])

        self.input_scaler_gpu = (
            torch.tensor(self.input_scaler[0], dtype=torch.float32, device=device),
            torch.tensor(self.input_scaler[1], dtype=torch.float32, device=device),
        )
        self.output_scaler_gpu = (
            torch.tensor(self.output_scaler[0], dtype=torch.float32, device=device),
            torch.tensor(self.output_scaler[1], dtype=torch.float32, device=device),
        )

    @staticmethod
    def _fit_minmax(arr):
        min_v = arr.min(axis=0)
        range_v = arr.max(axis=0) - min_v
        range_v[range_v == 0] = 1.0
        return min_v, range_v

    def scale(self, x, scaler):
        min_v, range_v = scaler
        return (x - min_v) / range_v

    def inverse_scale_gpu(self, x, scaler_gpu):
        min_v, range_v = scaler_gpu
        return x * range_v + min_v

    def __getitem__(self, idx):
        x = self.scale(self.inputs[idx], self.input_scaler)
        y = self.scale(self.outputs[idx], self.output_scaler)
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )

    def __len__(self):
        return self.N

# =========================================================
# Transformer Encoder
# =========================================================
class ProteinTransformer(nn.Module):
    def __init__(self):
        super().__init__()

        self.h_proj = nn.Sequential(
            nn.Linear(1, config.d_proj),
            nn.LayerNorm(config.d_proj),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
        self.condition_proj = nn.Sequential(
            nn.Linear(1, config.d_proj),
            nn.LayerNorm(config.d_proj),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )

        self.joint_proj = nn.Linear(Config.d_proj, config.d_model)

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
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.d_h_head),
            nn.GELU(),
            nn.Linear(config.d_h_head, 1)
        )

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src):
        delta_rmsd = src[:, 0:1]
        delta_rg   = src[:, 1:2]
        h_feats    = src[:, 2:]

        tokens = [
            self.condition_proj(delta_rmsd),
            self.condition_proj(delta_rg),
        ]
        for i in range(h_feats.shape[1]):
            tokens.append(self.h_proj(h_feats[:, i:i+1]))

        x = self.joint_proj(torch.stack(tokens, dim=1))
        encoded = self.transformer(x)

        h_tokens = encoded[:, 2:, :]
        return self.h_head(h_tokens).squeeze(-1)

# =========================================================
# Loss / Metrics
# =========================================================
class EnhancedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.weight = 1.0

    def forward(self, pred, target):
        h_loss = self.mse(pred, target)
        return self.weight * h_loss, h_loss.item()

def compute_accuracy(pred, target, dataset):
    pred_inv = dataset.inverse_scale_gpu(pred, dataset.output_scaler_gpu)
    tgt_inv  = dataset.inverse_scale_gpu(target, dataset.output_scaler_gpu)
    acc_1 = ((pred_inv - tgt_inv).abs() < 0.1).float().mean().item()
    acc_2 = ((pred_inv - tgt_inv).abs() < 0.2).float().mean().item()
    acc_3 = ((pred_inv - tgt_inv).abs() < 0.3).float().mean().item()
    acc_4 = ((pred_inv - tgt_inv).abs() < 0.4).float().mean().item()
    acc_5 = ((pred_inv - tgt_inv).abs() < 0.5).float().mean().item()
    return [acc_1, acc_2, acc_3, acc_4, acc_5]

# =========================================================
# Scheduler
# =========================================================
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps, max_lr):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.max_lr = max_lr
        self.step_count = 0

    def step(self):
        self.step_count += 1
        if self.step_count < self.warmup_steps:
            lr = self.max_lr * self.step_count / self.warmup_steps
        else:
            p = (self.step_count - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = 0.5 * self.max_lr * (1 + math.cos(math.pi * p))
        for g in self.optimizer.param_groups:
            g["lr"] = lr
        return lr

# =========================================================
# Train / Val Loop
# =========================================================
def run_epoch(model, loader, optimizer, criterion, scheduler, scaler, dataset, train=True):
    model.train() if train else model.eval()
    total_loss = total_h_loss = 0
    total_acc_1 = total_acc_2 = total_acc_3 = total_acc_4 = total_acc_5 = 0
    
    with torch.set_grad_enabled(train):
        for src, tgt in tqdm(loader, desc="Train" if train else "Val"):
            src, tgt = src.to(device), tgt.to(device)

            if train:
                optimizer.zero_grad()

            with autocast():
                pred = model(src)
                loss, h_loss = criterion(pred, tgt)

            if train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

            total_loss += loss.item()
            total_h_loss += h_loss
            acc_lis = compute_accuracy(pred, tgt, dataset)
            total_acc_1 += acc_lis[0]
            total_acc_2 += acc_lis[1]
            total_acc_3 += acc_lis[2]
            total_acc_4 += acc_lis[3]
            total_acc_5 += acc_lis[4]

    n = len(loader)
    return total_loss/n, total_h_loss/n, [total_acc_1/n, total_acc_2/n, total_acc_3/n, total_acc_4/n, total_acc_5/n]

# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    dataset = MappingProteinDataset(config.dataset_path,config.train_size,config.val_size)

    train_set, val_set = random_split(dataset,[config.train_size, config.val_size],generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_set, batch_size=config.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = ProteinTransformer().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = EnhancedLoss()
    scaler = GradScaler()

    total_steps = len(train_loader) * config.num_epochs
    scheduler = WarmupCosineScheduler(optimizer, config.warmup_steps, total_steps, config.learning_rate)

    os.makedirs(os.path.dirname(config.log_path), exist_ok=True)
    os.makedirs(os.path.dirname(config.model_save_path), exist_ok=True)

    with open(config.log_path, "w") as f:
        f.write("Epoch,TL,THL,TA1,TA2,TA3,TA4,TA5,VL,VHL,VA1,VA2,VA3,VA4,VA5,Time\n")
    # TL -- 训练集loss
    # THL -- 训练集的H 的loss
    # VL -- 验证集的 loss
    # TA{i} -- 训练集准确率，特征命中阈值为i/10 

    best_val = float("inf")

    for epoch in range(config.num_epochs):
        t0 = time.time()
        tr = run_epoch(model, train_loader, optimizer, criterion, scheduler, scaler, dataset, True)
        va = run_epoch(model, val_loader, optimizer, criterion, scheduler, scaler, dataset, False)
        dt = time.time() - t0

        with open(config.log_path, "a") as f:
            f.write(f"{epoch+1},{tr[0]:.6f},{tr[1]:.6f},{tr[2][0]:.6f},{tr[2][1]:.6f},{tr[2][2]:.6f},{tr[2][3]:.6f},{tr[2][4]:.6f},"
                    f"{va[0]:.6f},{va[1]:.6f},{va[2][0]:.6f},{va[2][1]:.6f},{va[2][2]:.6f},{va[2][3]:.6f},{va[2][4]:.6f},{dt:.2f}\n")

        print(f"\nEpoch {epoch+1}/{config.num_epochs} | Time {dt:.1f}s")
        print(f"Train: loss {tr[0]:.4f}, acc1 {tr[2][0]:.2%},acc2 {tr[2][1]:.2%},acc3 {tr[2][2]:.2%},acc4 {tr[2][3]:.2%},acc5 {tr[2][4]:.2%}")
        print(f"Val  : loss {va[0]:.4f}, acc1 {va[2][0]:.2%},acc2 {va[2][1]:.2%},acc3 {va[2][2]:.2%},acc4 {va[2][3]:.2%},acc5 {va[2][4]:.2%}\n")

        if va[0] < best_val:
            best_val = va[0]
            checkpoint = {
                "model_state": model.state_dict(),
                "input_scaler": {
                    "min": dataset.input_scaler[0],
                    "range": dataset.input_scaler[1],
                },
                "output_scaler": {
                    "min": dataset.output_scaler[0],
                    "range": dataset.output_scaler[1],
                },
                "config": vars(config)
            }

            torch.save(checkpoint, config.model_save_path)
            print("✅ Saved best model\n")
            # torch.save(model.state_dict(), config.model_save_path)

