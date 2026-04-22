import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from Bio.PDB import PDBParser
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

# class ProteinDataset(Dataset):
#     def __init__(self, pdb_dir):
#         self.pdb_dir = pdb_dir
#         self.file_list = sorted([f for f in os.listdir(pdb_dir) if f.endswith('.pdb')])
        
#         self.all_coords = []
#         for pdb_file in tqdm(self.file_list, desc="Collecting coordinates for normalization"):
#             coords = self._load_pdb_coords(os.path.join(pdb_dir, pdb_file))
#             if coords is not None:
#                 self.all_coords.append(coords)
        
#         self.scaler = MinMaxScaler()
#         all_coords_flat = np.concatenate(self.all_coords)
#         self.scaler.fit(all_coords_flat)
        
#         self.data = []
#         for pdb_file in tqdm(self.file_list, desc="Loading and normalizing PDB files"):
#             coords = self._load_pdb_coords(os.path.join(pdb_dir, pdb_file))
#             if coords is not None:
#                 normalized_coords = self.scaler.transform(coords)
#                 self.data.append(normalized_coords)
        
#         self.data = np.array(self.data)
#         print(f"Successfully loaded {len(self.data)} structures")

class ProteinDataset(Dataset):
    def __init__(self, pdb_dir, index_txt):
        self.pdb_dir = pdb_dir
        with open(index_txt, 'r') as f:
            lines = f.readlines()

        codes = [line.strip().split()[0] for line in lines[1:] if line.strip()]
        self.file_list = []
        for code in codes:
            pdb_file = f"aligned_{code}.pdb"
            pdb_path = os.path.join(pdb_dir, pdb_file)
            if os.path.exists(pdb_path):
                self.file_list.append(pdb_file)
            else:
                print(f"Warning: {pdb_file} not found, skipping.")

        print(f"Total valid PDB files: {len(self.file_list)}")

        self.all_coords = []
        for pdb_file in tqdm(self.file_list, desc="Collecting coordinates for normalization"):
            coords = self._load_pdb_coords(os.path.join(pdb_dir, pdb_file))
            if coords is not None:
                self.all_coords.append(coords)

        self.scaler = MinMaxScaler()
        all_coords_flat = np.concatenate(self.all_coords)
        self.scaler.fit(all_coords_flat)

        self.data = []
        for pdb_file in tqdm(self.file_list, desc="Loading and normalizing PDB files"):
            coords = self._load_pdb_coords(os.path.join(pdb_dir, pdb_file))
            if coords is not None:
                normalized_coords = self.scaler.transform(coords)
                self.data.append(normalized_coords)

        self.data = np.array(self.data)
        print(f"Successfully loaded {len(self.data)} structures")

    def _load_pdb_coords(self, pdb_path):
        parser = PDBParser(QUIET=True)
        try:
            structure = parser.get_structure('protein', pdb_path)
            coords = []
            
            for model in structure:
                for chain in model:
                    for residue in chain:
                        
                        # Extract N, CA, C
                        residue_atoms = {'N': None, 'CA': None, 'C': None}
                        oxygen_coord = None

                        for atom in residue:
                            atom_name = atom.name.strip()
                            
                            if atom_name in residue_atoms:
                                residue_atoms[atom_name] = atom.coord
                            
                            # Flexible oxygen selection
                            if atom_name in ['O', 'OT1', 'OXT']:
                                if oxygen_coord is None:
                                    oxygen_coord = atom.coord

                        if all(residue_atoms[k] is not None for k in ['N', 'CA', 'C']) and oxygen_coord is not None:
                            coords.extend([
                                residue_atoms['N'],
                                residue_atoms['CA'],
                                residue_atoms['C'],
                                oxygen_coord
                            ])
                        else:
                            missing = [k for k, v in residue_atoms.items() if v is None]
                            if oxygen_coord is None:
                                missing.append('O/OT1/OXT')
                            print(f"Missing atoms {missing} in {os.path.basename(pdb_path)} residue {residue.id}")
                            return None
            
            if len(coords) == 0:
                return None
                
            return np.array(coords)

        except Exception as e:
            print(f"Error loading {os.path.basename(pdb_path)}: {str(e)}")
            return None

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx]).flatten()  #Flatten into a 1D vector
    
    def get_scaler(self):
        return self.scaler

class BN_Layer(nn.Module):
    def __init__(self, dim_z, tau, mu=True):
        super(BN_Layer, self).__init__()
        self.dim_z = dim_z

        self.tau = torch.tensor(tau)  # tau : float in range (0,1)
        self.theta = torch.nn.Parameter(torch.tensor(0.5), requires_grad=True)

        self.bn = nn.BatchNorm1d(dim_z, affine=True)
        self.bn.bias.requires_grad = False

        self.mu = mu

    def forward(self, x):  # x: (batch_size, dim_z)
        if self.mu:
            gamma = torch.sqrt(self.tau + (1 - self.tau) * torch.sigmoid(self.theta))
        else:
            gamma = torch.sqrt((1 - self.tau) * torch.sigmoid((-1) * self.theta))

        gamma = nn.Parameter(gamma.expand_as(self.bn.weight))

        self.bn.weight = gamma

        x = self.bn(x)
        return x

# class BN_Layer(nn.Module):
#     def __init__(self, dim_z, tau, mu=True):
#         super(BN_Layer, self).__init__()

#         self.dim_z = dim_z
#         self.mu = mu
#         self.register_buffer("tau", torch.tensor(float(tau)))         # tau 应该随模型保存但不训练
#         self.theta = nn.Parameter(torch.tensor(0.5))         # 可学习参数
#         self.bn = nn.BatchNorm1d(dim_z, affine=False)  # 使用 BN 但不使用 affine，因为 gamma 由我们控制

#     def _compute_gamma(self):
#         if self.mu:
#             gamma = torch.sqrt(
#                 self.tau + (1.0 - self.tau) * torch.sigmoid(self.theta)
#             )
#         else:
#             gamma = torch.sqrt(
#                 (1.0 - self.tau) * torch.sigmoid(-self.theta)
#             )
#         gamma = gamma.expand(self.dim_z)         # 扩展为每个维度一个 gamma
#         return gamma

#     def forward(self, x):
#         """
#         x: shape (batch_size, dim_z)
#         """

#         gamma = self._compute_gamma()

#         # 使用 functional BN 以允许自定义 gamma
#         x = F.batch_norm(
#             x,
#             self.bn.running_mean,
#             self.bn.running_var,
#             weight=gamma,
#             bias=None,
#             training=self.bn.training,
#             momentum=self.bn.momentum,
#             eps=self.bn.eps,
#         )

#         return x

class VAEEncoder(nn.Module):
    def __init__(self, input_size, hidden_sizes, latent_size, tau=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.bn1 = nn.BatchNorm1d(hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.bn2 = nn.BatchNorm1d(hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.bn3 = nn.BatchNorm1d(hidden_sizes[2])
        self.fc4 = nn.Linear(hidden_sizes[2], hidden_sizes[3])
        self.bn4 = nn.BatchNorm1d(hidden_sizes[3])
        self.fc5 = nn.Linear(hidden_sizes[3], latent_size * 2)

        # 使用 BN_Layer
        self.bn_mu = BN_Layer(latent_size, tau, mu=True)
        self.bn_log_var = BN_Layer(latent_size, tau, mu=False)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.fc1(x)), negative_slope=0.1)
        x = F.leaky_relu(self.bn2(self.fc2(x)), negative_slope=0.1)
        x = F.leaky_relu(self.bn3(self.fc3(x)), negative_slope=0.1)
        x = F.leaky_relu(self.bn4(self.fc4(x)), negative_slope=0.1)
        x = self.fc5(x)
        mu, log_var = x.split(x.size(1) // 2, dim=1)

        mu = self.bn_mu(mu)
        log_var = self.bn_log_var(log_var)

        return mu, log_var

class VAEDecoder(nn.Module):
    def __init__(self, latent_size, hidden_sizes, output_size):
        super().__init__()
        self.fc1 = nn.Linear(latent_size, hidden_sizes[3])
        self.bn1 = nn.BatchNorm1d(hidden_sizes[3])
        self.fc2 = nn.Linear(hidden_sizes[3], hidden_sizes[2])
        self.bn2 = nn.BatchNorm1d(hidden_sizes[2])
        self.fc3 = nn.Linear(hidden_sizes[2], hidden_sizes[1])
        self.bn3 = nn.BatchNorm1d(hidden_sizes[1])
        self.fc4 = nn.Linear(hidden_sizes[1], hidden_sizes[0])
        self.bn4 = nn.BatchNorm1d(hidden_sizes[0])
        self.fc5 = nn.Linear(hidden_sizes[0], output_size)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.fc1(x)), negative_slope=0.1)
        x = F.leaky_relu(self.bn2(self.fc2(x)), negative_slope=0.1)
        x = F.leaky_relu(self.bn3(self.fc3(x)), negative_slope=0.1)
        x = F.leaky_relu(self.bn4(self.fc4(x)), negative_slope=0.1)
        x = torch.tanh(self.fc5(x)) # self.fc5(x)
        return x

class VAE(nn.Module):
    def __init__(self, input_size, hidden_sizes, latent_size, tau=0.5):
        super(VAE, self).__init__()
        self.encoder = VAEEncoder(input_size, hidden_sizes, latent_size, tau)
        self.decoder = VAEDecoder(latent_size, hidden_sizes, input_size)

    def forward(self, x):
        mu, log_var = self.encoder(x)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + std * eps
        recon = self.decoder(z)
        return recon, mu, log_var

def vae_loss(recon_x, x, mu, logvar, beta=0.5):
    MSE = F.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = 1.2 * MSE + beta * KLD
    return MSE, KLD, total_loss

def train_vae(model, dataloader, optimizer, device, epochs=100):
    model.train()
    best_recon_loss = float('inf')
    
    with open("loss/loss_vae.txt", "w") as f:
        f.write("epoch\tloss\trecon_loss\tkld_loss\n")
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
        
        for epoch in range(epochs):
            total_loss = 0
            recon_loss = 0
            kld_loss = 0
            
            for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                batch = batch.to(device)
                optimizer.zero_grad()
                
                recon_batch, mu, logvar = model(batch)
                recon_loss_val, kld_val, loss = vae_loss(recon_batch, batch, mu, logvar)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                recon_loss += recon_loss_val.item()
                kld_loss += kld_val.item()
            
            num_samples = len(dataloader.dataset)
            avg_loss = total_loss / num_samples
            avg_recon = recon_loss / num_samples
            avg_kld = kld_loss / num_samples
            
            scheduler.step(avg_recon)
            
            if avg_recon < best_recon_loss:
                best_recon_loss = avg_recon
                torch.save(model.state_dict(), "models/vae.pth")
            
            f.write(f"{epoch+1}\t{avg_loss:.6f}\t{avg_recon:.6f}\t{avg_kld:.6f}\n")
            
            print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, "
                  f"Recon: {avg_recon:.4f}, KLD: {avg_kld:.4f}, LR: {optimizer.param_groups[0]['lr']:.2e}")

def train():
    os.makedirs("loss", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    pdb_dir = "/mnt/sto3/wenyanbing/DA2-GRASP/Trp-cage/DATASET/aligned"
    index_txt = "/mnt/sto3/wenyanbing/DA2-GRASP/Trp-cage/DATASET/rmsd_rg_em_bb.txt"
    
    batch_size = 64 
    latent_size = 8
    hidden_sizes = [512, 256, 128, 64]
    learning_rate = 1e-4
    epochs = 100
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    
    print("Loading dataset...")
    dataset = ProteinDataset(pdb_dir, index_txt)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    sample = next(iter(dataloader))
    input_size = sample.shape[1]
    print(f"Detected input size: {input_size}")
    
    print("Initializing model...")
    model = VAE(input_size, hidden_sizes, latent_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    print("Starting training...")
    train_vae(model, dataloader, optimizer, device, epochs)
    
    scaler_path = os.path.join("models", "vae-scaler.pkl")
    import joblib
    joblib.dump(dataset.get_scaler(), scaler_path)
    print(f"Scaler saved to {scaler_path}")

if __name__ == "__main__":
    train()