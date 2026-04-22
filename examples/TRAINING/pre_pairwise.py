import os
import json
import random
import joblib
import torch
import numpy as np
import prody as pd
from tqdm import tqdm
from train_vae import VAE

# ============================================================
# 1. VAE configurations 
# ============================================================
config_vae = {
    'pdb_dir': ".././DATASET/aligned",
    'rmsd_rg_file': ".././DATASET/git_rmsd_rg_em_examples.txt",
    'vae_model': 'models/vae.pth',
    'scaler': 'models/vae-scaler.pkl',
    'output_json': 'data-mapping/pre_pairwise.json',
    'dims_layer':[512, 256, 128, 64],
    'latent_size': 8,
    'input_size': 240  # 80 atoms × 3
}

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

model = VAE(config_vae['input_size'], config_vae['dims_layer'], config_vae['latent_size']).to(device)
model.load_state_dict(torch.load(config_vae['vae_model'], map_location=device))
model.eval()

scaler = joblib.load(config_vae['scaler'])


# ============================================================
# 2. PDB → low dimensional representations
# ============================================================
def get_backbone_atoms(structure):
    atoms = []
    residues = list(structure.iterResidues())
    for i, res in enumerate(residues):
        o_name = 'OT1' if i == len(residues) - 1 else 'O'
        for atom in res:
            if atom.getName() in ['N', 'CA', 'C', o_name]:
                atoms.append(atom)
    return atoms


@torch.no_grad()
def encode_pdb(pdb_path):
    structure = pd.parsePDB(pdb_path)
    atoms = get_backbone_atoms(structure)
    coords = np.array([a.getCoords() for a in atoms])

    flat = coords.reshape(-1)
    norm = scaler.transform(flat.reshape(-1, 3))
    tensor = torch.FloatTensor(norm.reshape(1, -1)).to(device)

    mu, logvar = model.encoder(tensor)
    return mu.cpu().numpy()[0], logvar.cpu().numpy()[0]


def build_pre_pairwise():
    rmsd_rg = {}
    with open(config_vae['rmsd_rg_file']) as f:
        next(f)
        for line in f:
            code, rmsd, rg = line.split()
            rmsd_rg[code] = {'RMSD': float(rmsd), 'Rg': float(rg)}

    results = {}
    for code in tqdm(rmsd_rg, desc="Encoding PDBs"):
        pdb = os.path.join(config_vae['pdb_dir'], f"aligned_{code}.pdb")
        if not os.path.exists(pdb):
            continue

        mu, logvar = encode_pdb(pdb)
        results[code] = {
            'code': code,
            'data': {
                'RMSD': rmsd_rg[code]['RMSD'],
                'Rg': rmsd_rg[code]['Rg'],
                'Mu': mu.tolist(),
                'Logvar': logvar.tolist()
            }
        }

    os.makedirs(os.path.dirname(config_vae['output_json']), exist_ok=True)
    with open(config_vae['output_json'], 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Saved {len(results)} structures → {config_vae['output_json']}")

build_pre_pairwise()
