import os
import json
import torch
import joblib
import prody as pd
import subprocess
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter

from utils import calc_props_start
from utils import find_min_cvs_grad
from utils import calculate_gradients_distance
from utils import calculate_structures_properties

from train_vae import VAE
from train_mapping import ProteinTransformer

## usage ##
# nohup python da2_grasp.py -te 0 -m "u2f" -ss "[0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6]" -T 50 -Nd 8 > da2_grasp_10_u2f.log 2>&1 &

## Then ##
# nohup python da2_grasp.py -te 350 -m "u2f_reverse" -ss "[0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6]" -T 50 -Nd 8 > da2_grasp_10_u2f_reverse.log 2>&1 &

import argparse
import ast
import numbers

def parse_args():
    parser = argparse.ArgumentParser(description="DA2-GRASP Sampling Script")

    # m: mode for sampling
    parser.add_argument(
        '--tzero', '-te',
        type=int,
        required=True,
        help='Tzero for each trajectory, e.g., "0"'
    )

    # m: mode for sampling
    parser.add_argument(
        '--mode', '-m',
        type=str,
        required=True,
        help='Sampling mode for each trajectory, e.g., "u2f"'
    )

    # ss: list for step size
    parser.add_argument(
        '--step_sizes', '-ss',
        type=str,
        required=True,
        help='List of step sizes for each trajectory, e.g., "[1,2,3,4]"'
    )

    # T: Total number of step for each Traj.
    parser.add_argument(
        '--total_steps', '-T',
        type=int,
        required=True,
        help='Total number of sampling steps per trajectory (positive integer)'
    )

    # Nd: Number of transition directions
    parser.add_argument(
        '--num_directions', '-Nd',
        type=int,
        required=True,
        help='Number of candidate directions for each transition (positive integer)'
    )

    args = parser.parse_args()

    try:
        step_sizes = ast.literal_eval(args.step_sizes)

        if not isinstance(step_sizes, list):
            raise ValueError

        # ✅ 允许任意数值类型（int / float / 正负 / 0）
        if not all(isinstance(x, numbers.Number) for x in step_sizes):
            raise ValueError

    except Exception:
        raise ValueError(
            'Argument -ss/--step_sizes must be a list of numbers, e.g., "[1, -2, 0.5, 3]"'
        )
        
    if args.total_steps <= 0:
        raise ValueError("Argument -T/--total_steps must be a positive integer")

    if args.num_directions <= 0:
        raise ValueError("Argument -Nd/--num_directions must be a positive integer")

    args.step_sizes = step_sizes

    return args


args = parse_args()

mode_sam = args.mode
Step_size_lis = args.step_sizes
Num_step_per_traj = args.total_steps
Number_directions = args.num_directions
t_zero = args.tzero

# | Argument                   | Type        | Description                                                                                                                                                                                     |
# | -------------------------- | ----------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
# | `--mode` / `-m`            | string      | sampling mode, u2f, or "u2f_reverse".                                                                                                                             |
# | `--step_sizes` / `-ss`     | list of int | A list specifying the step size for each trajectory. The length of the list defines the number of trajectories, and each element controls the sampling stride for the corresponding trajectory. |
# | `--total_steps` / `-T`     | int         | Total number of sampling steps performed for each trajectory.                                                                                                                                   |
# | `--num_directions` / `-Nd` | int         | Number of candidate directions generated at each transition step for conformational exploration.                                                                                                |


num_vae = 10
# output_dir = f"results_{num_vae}_{mode_sam}"
output_dir = f"results"

config = {
    'init_pdb_u2f': "./common/pdbs/2jof_aligned_082709.pdb",
    'init_pdb_u2f_energy': -9.04746456119297e+02,
    'end_state_u2f': 0,

    'init_pdb_u2f_reverse': "./common/pdbs/2jof_aligned_028182.pdb",
    'init_pdb_u2f_energy_reverse': -2.66406823431145e+02,
    'end_state_u2f_reverse': 20,

    'template_pdb': "./common/pdbs/2jof_template.pdb",
                                  
    'ref_pdb': "./common/pdbs/2jof_ref.pdb",
    'mode': mode_sam,

    'vae_model': f".././TRAINING/models/vae.pth",
    'vae_scaler': f".././TRAINING/models/vae-scaler.pkl",
    'vae_layers_dim':[512, 256, 128, 64],
    'mapping_model': f".././TRAINING/models/mapping-{num_vae}-43.pth",
    'output_cluster': [f"{output_dir}/cluster_backbone", f"{output_dir}/cluster_full", f"{output_dir}/cluster_opt"],
    'num_latent_structures': 100,
    'latent_size': 8,
    'input_size': 240 # 20 resi * 4 atoms * 3 coordinates
}

os.makedirs(config['output_cluster'][0], exist_ok=True)
os.makedirs(config['output_cluster'][1], exist_ok=True)
os.makedirs(config['output_cluster'][2], exist_ok=True)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def load_vae(device):
    vae_model = VAE(config['input_size'], config['vae_layers_dim'], config['latent_size']).to(device)
    vae_model.load_state_dict(torch.load(config['vae_model'], map_location=device))
    vae_model.eval()
    return vae_model

def load_mapping_scaler(device, mapping_and_scaler):
    mapping_model = ProteinTransformer().to(device)
    mapping_model.load_state_dict(mapping_and_scaler["model_state"])
    mapping_model.eval()
    return mapping_model

vae_scc = load_vae(device)
vae_scaler = joblib.load(config['vae_scaler'])

mapping_and_scaler = torch.load(config['mapping_model'], map_location=device)
h_in_min  = torch.tensor(mapping_and_scaler["input_scaler"]["min"], dtype=torch.float32, device=device)
h_in_rng  = torch.tensor(mapping_and_scaler["input_scaler"]["range"], dtype=torch.float32, device=device)
h_out_min = torch.tensor(mapping_and_scaler["output_scaler"]["min"], dtype=torch.float32, device=device)
h_out_rng = torch.tensor(mapping_and_scaler["output_scaler"]["range"], dtype=torch.float32, device=device)
mapping = load_mapping_scaler(device, mapping_and_scaler)

def h_scale_input(x):
    return (x - h_in_min) / h_in_rng

def h_inverse_output(y):
    return y * h_out_rng + h_out_min

def get_backbone_atoms(structure, is_source=False):
    atoms = []
    residues = list(structure.iterResidues()) 
    for i, res in enumerate(residues):
        oxygen_name = 'OT1' if is_source and i == len(residues)-1 else 'O'
        for atom in res:
            if atom.getName() in ['N', 'CA', 'C', oxygen_name]:
                atoms.append(atom)
    return atoms

def get_source_coords(pdb_path):
    source = pd.parsePDB(pdb_path, QUIET=True)
    source_atoms = get_backbone_atoms(source, is_source=True)
    source_coords = np.array([atom.getCoords() for atom in source_atoms])
    return source_coords

def h_mapping_ij(delta_cvs, coords_3d):
    delta_cv_s = torch.as_tensor(delta_cvs, dtype=torch.float32).to(device)
    with torch.no_grad():
        flat_coords = coords_3d.reshape(-1)
        norm_coords = vae_scaler.transform(flat_coords.reshape(-1, 3))
        tensor_coords = torch.FloatTensor(norm_coords.reshape(1,-1)).to(device)
        mu_i, logvar_i = vae_scc.encoder(tensor_coords)
    delta_cvij_hi = torch.cat([delta_cv_s.unsqueeze(0), mu_i, logvar_i], dim=1) 
    delta_cvij_hi_scaled = h_scale_input(delta_cvij_hi)
    
    with torch.no_grad():
        h_j = mapping(delta_cvij_hi_scaled) #.unsqueeze(0)
    hj_scaled = h_inverse_output(h_j)

    mu_j = hj_scaled[:, 0:config['latent_size']] 
    logvar_j = hj_scaled[:, config['latent_size']:config['latent_size']*2]
    return mu_i, logvar_i, mu_j, logvar_j
    
def cluster_protein_structures(matrix_list, n_clusters=1): 
    reference = matrix_list[0]
    rmsd_values = []
    for matrix in matrix_list:
        squared_diff = np.sum((matrix - reference)**2, axis=1)
        rmsd = np.sqrt(np.mean(squared_diff))
        rmsd_values.append(rmsd)
 
    X = np.array(rmsd_values).reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=3)
    kmeans.fit(X)
    
    cluster_labels = kmeans.labels_
    cluster_counts = Counter(cluster_labels)
    largest_cluster = cluster_counts.most_common(1)[0][0]

    cluster_indices = np.where(cluster_labels == largest_cluster)[0]
    cluster_rmsds = X[cluster_indices].flatten()
    
    cluster_mean = np.mean(cluster_rmsds)
    closest_idx = cluster_indices[np.argmin(np.abs(cluster_rmsds - cluster_mean))]
    
    return closest_idx + 1, matrix_list[closest_idx] #closest_idx + 1是为了与文件索引一致

def get_new_backbone_and_sidechain(delta_cvs, coords_t, direct_id, n_latent=10, t=1):

    mu_i, logvar_i, mu_j, logvar_j = h_mapping_ij(delta_cvs, coords_t)
    Latent_space = {
        'Step': t,
        'Direction': direct_id + 1,
        'Input': {
            'delta_cv': delta_cvs.tolist(),
            'h_mu': mu_i.cpu().numpy().tolist(),
            'h_logvar': logvar_i.cpu().numpy().tolist()
        },
        'Output': {
            'h_mu': mu_j.cpu().numpy().tolist(),
            'h_logvar': logvar_j.cpu().numpy().tolist(),
        }
    }

    backbone_all_coors_t = []
    for _ in range(n_latent):
        with torch.no_grad():
            z = mu_j + torch.randn_like(mu_j) * torch.exp(0.5 * logvar_j)
            generated = vae_scc.decoder(z).cpu().numpy()
        
        backbone_coords = vae_scaler.inverse_transform(generated.reshape(-1, 3))
        backbone_all_coors_t.append(backbone_coords)
    return Latent_space, backbone_all_coors_t

def sample_mode(mode_sam ='u2f'):
    if mode_sam == 'u2f':
        end_rg = config['end_state_u2f']
        energy_ori = config['init_pdb_u2f_energy']
        coords_ori = get_source_coords(config['init_pdb_u2f'])
        rmsd_ori, rg_ori = calc_props_start(config['init_pdb_u2f'], config['ref_pdb'])
        distance_to_end = np.sqrt((rg_ori - end_rg)**2)
        
    elif mode_sam == 'u2f_reverse':
        end_rg = config['end_state_u2f_reverse']
        energy_ori = config['init_pdb_u2f_energy_reverse']
        coords_ori = get_source_coords(config['init_pdb_u2f_reverse'])
        rmsd_ori, rg_ori = calc_props_start(config['init_pdb_u2f_reverse'], config['ref_pdb'])
        distance_to_end = np.sqrt((rg_ori - end_rg)**2)
    else:
        print(f'Error: No such sample mode {mode_sam}')
    return energy_ori, coords_ori, rmsd_ori, rg_ori, distance_to_end, end_rg

def normalize_vector(v):
    norm = np.linalg.norm(v)
    return np.array(v) / norm if norm > 0 else np.array(v)


# Number_directions = args.num_directions

# wRg = 1
# DIRECTION_VECTORS = [[0, 1 * wRg], [1/np.sqrt(2), 1/np.sqrt(2) * wRg], [1, 0], [1/np.sqrt(2), -1/np.sqrt(2) * wRg],[0, -1 * wRg], 
#                      [-1/np.sqrt(2), -1/np.sqrt(2) * wRg], [-1, 0], [-1/np.sqrt(2), 1/np.sqrt(2) * wRg]]
# directions = [normalize_vector(v) for v in DIRECTION_VECTORS]



if __name__ == "__main__":

    All_Latent_data = []
    All_grad_distance = []
    Best_cvs_path = []
    json_dir = f"{output_dir}/json"
    opt_dir = config['output_cluster'][2] + "/" + "opt"
    os.makedirs(opt_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)

    template = pd.parsePDB(config['template_pdb'], QUIET=True)
    template_atoms = get_backbone_atoms(template)                 
                                                        
    T = int(Num_step_per_traj) # 50                                              
    dlis = Step_size_lis  # [0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6]                  
    base_angles = np.linspace(0, 2 * np.pi, num=Number_directions, endpoint=False)
    
    ######## Backward grid search algorithm ##########
    intial_step = 0                                        
    step_search_factor = 0.2                                    
    Nbgs = 50 
    depth_max = step_search_factor * Nbgs               
    ###################################
    
    for di in range(len(dlis)):
        stepsize = dlis[di]
        # delta_cvs_list = [np.array(v) * stepsize for v in directions]
        step_lis = [t_zero + t + T * di for t in range(T)]
        energy_ori, coords_ori, rmsd_ori, rg_ori, distance_to_end, rg_end = sample_mode(config['mode'])
        
        for t in step_lis:
            cluster_idx_lis = [] 
            cluster_backbones = []
            all_backbones = []
        
            random_offset = np.random.uniform(0, 2 * np.pi)                                 
            angles_vectors = (base_angles + random_offset) % (2 * np.pi)
            NORMOLIZED_DIRECTION_VECTORS = [[np.cos(theta), np.sin(theta)] for theta in angles_vectors]
            delta_cvs_list = [np.array(v) * stepsize for v in NORMOLIZED_DIRECTION_VECTORS]

            for direction, delt_CV in enumerate(delta_cvs_list):
                Latent_space, backbone_all_coors_t = get_new_backbone_and_sidechain(delt_CV, coords_ori, direction, config['num_latent_structures'], t)
                all_backbones.append(backbone_all_coors_t)
                All_Latent_data.append(Latent_space) 
                cluster_idx, cluster_backbone = cluster_protein_structures(backbone_all_coors_t) 
                cluster_idx_lis.append(cluster_idx)
                cluster_backbones.append(cluster_backbone)

            os.makedirs(config['output_cluster'][0]+"/"+str(t), exist_ok=True)
            os.makedirs(config['output_cluster'][1]+"/"+str(t), exist_ok=True)
            for i in range(len(cluster_idx_lis)):
                for j, atom in enumerate(template_atoms):
                    atom.setCoords(cluster_backbones[i][j])

                cluster_backbone_path = f"{config['output_cluster'][0]}/{str(t)}/backbone_{i+1}.pdb" 
                cluster_sidechain_path = f"{config['output_cluster'][1]}/{str(t)}/sidechain_{i+1}.pdb"
                pd.writePDB(cluster_backbone_path, template)
                subprocess.run(f"Scwrl4 -i {cluster_backbone_path} -o {cluster_sidechain_path} -h -t > ./{output_dir}/sidechain-{config['mode']}.log", shell=True, check=True)

            full_pdb_path = f"{config['output_cluster'][1]}/{str(t)}"
            full_opt_path = f"{config['output_cluster'][2]}/{str(t)}_pdb"
            full_txt_path = f"{config['output_cluster'][2]}/txt"
            energy_opt_path = f"{config['output_cluster'][2]}/txt/output_{str(t)}.txt"
            
            os.makedirs(full_opt_path, exist_ok=True)
            os.makedirs(full_txt_path, exist_ok=True)
            log_minim_ = f"{output_dir}/minim_{mode_sam}"
            os.makedirs(log_minim_, exist_ok=True)

            full_pdb_files = [f for f in os.listdir(full_pdb_path) if f.endswith('.pdb')]
            subprocess.run(f"./minim.sh -i {full_pdb_path} -o {full_opt_path} -x {full_txt_path} -s {str(t)} -l {log_minim_} -n {Number_directions} > ./{output_dir}/minim_{mode_sam}.log 2>&1", shell=True, check=True)
            cvs_opt_path = f"{config['output_cluster'][2]}/txt/output_opt_{str(t)}.txt"

            calculate_structures_properties(config['ref_pdb'], full_pdb_path, full_opt_path, energy_opt_path, cvs_opt_path)
            grad_distance_t = calculate_gradients_distance(rmsd_ori, rg_ori, rg_end, energy_ori, t, cvs_opt_path)
            All_grad_distance.append(grad_distance_t)

            best_cvs_path_t = find_min_cvs_grad(grad_distance_t, distance_to_end, stepsize, intial_step, step_search_factor, depth_max)
            best_cvs_path_t["step"] = t
            Best_cvs_path.append(best_cvs_path_t)
            distance_to_end = best_cvs_path_t["distance_to_end"]

            protein_ori = f"{config['output_cluster'][2]}/{str(t)}_pdb/opt_{best_cvs_path_t['best_direction']}.pdb"
            protein_ori_cp = f"{opt_dir}/opt_{str(t)}_{best_cvs_path_t['best_direction']}.pdb"
            
            subprocess.run(f"cp {protein_ori} {protein_ori_cp}", shell=True, check=True)
            coords_ori = get_source_coords(protein_ori)
            energy_ori = best_cvs_path_t["Energy"]
            rmsd_ori, rg_ori = best_cvs_path_t["CVs"]

        
    Latent_path = os.path.join(json_dir, f"{config['mode']}_latent_space.json")
    with open(Latent_path, 'w') as f:
        json.dump(All_Latent_data, f, indent=4)

    grad_distance_path = os.path.join(json_dir, f"{config['mode']}_grad_distance.json")
    with open(grad_distance_path, 'w') as f:
        json.dump(All_grad_distance, f, indent=4)

    best_sample_path = os.path.join(json_dir, f"{config['mode']}_favorable_transition.json")
    with open(best_sample_path, 'w') as f:
        json.dump(Best_cvs_path, f, indent=4)
