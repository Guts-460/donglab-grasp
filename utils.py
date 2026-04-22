import torch
import torch.nn as nn

class b2dVAE(nn.Module):
    def __init__(self, input_dim=2, latent_dim=8):
        super(b2dVAE, self).__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.fc_mean = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mean(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar
    

class Config:
    input_dim = 19  # delta_x, delta_y, energy(3) + h(16)
    output_dim = 17  # energy(1) + h(16)
    d_model = 256
    num_layers = 6
    dim_feedforward = 1024
    dropout = 0.1
    batch_size = 256
    num_epochs = 100
    learning_rate = 3e-4
    warmup_steps = 4000
    grad_clip = 1.0
    early_stop_patience = 15
    model_save_path = "models/m-s2s-b2d.pth"
    combined_loss_path = "loss/loss-s2s-b2d.txt"
    random_sample_size = 100000

config = Config()

class b2dS2S(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.input_proj = nn.Sequential(
            nn.Linear(config.input_dim, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, config.d_model)
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=8,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers
        )
        
        self.output_proj = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, config.output_dim)
        )
        
    def forward(self, src):
        src = self.input_proj(src)  # [B, d_model]
        src = src.unsqueeze(1)  # [B, 1, d_model]
        encoded = self.transformer_encoder(src)  # [B, 1, d_model]
        output = self.output_proj(encoded.squeeze(1))  # [B, output_dim]
        
        return output