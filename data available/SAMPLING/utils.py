import os
import warnings
from Bio.PDB.PDBExceptions import PDBConstructionWarning

warnings.filterwarnings("ignore", category=PDBConstructionWarning)

from Bio import PDB
import numpy as np


################### Calculation of RMSD, Rg #########################
########## PDB ID in eight directions, Un_opt_RMSD, Un_opt_Rg, Opt_RMSD, Opt_Rg, Opt_Potential Energy ##############
import MDAnalysis as mda
from MDAnalysis.analysis import rms, align

def calculate_structures_properties(ref_pdb, unopt_dir, opt_dir, energy_file, output_path):
    ref = mda.Universe(ref_pdb)
    backbone = "protein and (name N or name CA or name C or name O or name OT1)"
    ref_atoms = ref.select_atoms(backbone)

    energy_data = {}
    directions = []
    with open(energy_file) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):  
                continue
                
            parts = line.split()
            if len(parts) < 2:
                print(f"警告: 第{line_num}行格式不正确，跳过: {line}")
                continue
                
            try:
                key = parts[0]
                value = float(parts[1])
                energy_data[key] = value
                directions.append(key)
            except ValueError:
                print(f"警告: 第{line_num}行数值格式错误，跳过: {line}")
                continue

    with open(output_path, 'w') as f:
        f.write("StructureID\tUnopt_RMSD\tUnopt_Rg\tOpt_RMSD\tOpt_Rg\tOpt_Energy\n")

    for fi in directions:
        sid = fi
        unopt = os.path.join(unopt_dir, f"sidechain_{sid}.pdb")
        urmsd, urg = calc_props(unopt, ref, backbone, ref_atoms)
        
        opt = os.path.join(opt_dir, f"opt_{sid}.pdb")
        ormsd, org = calc_props(opt, ref, backbone, ref_atoms)
        
        with open(output_path, 'a') as f:
            f.write(f"{sid}\t{urmsd:.4f}\t{urg:.4f}\t{ormsd:.4f}\t{org:.4f}\t{energy_data.get(sid, 'NA'):.4f}\n")

def calc_props(pdb, ref, selection, ref_atoms):
    mobile = mda.Universe(pdb)
    align.AlignTraj(mobile, ref, select=selection, in_memory=True).run()
    r = rms.rmsd(mobile.select_atoms(selection).positions, ref_atoms.positions, superposition=True)
    g = mobile.select_atoms(selection).radius_of_gyration()
    return r, g

######## Calculate the RMSD and Rg of the starting structure ##############
def calc_props_start(pdb, ref_pdb):
    ref = mda.Universe(ref_pdb)
    backbone = "protein and (name N or name CA or name C or name O or name OT1)"
    ref_atoms = ref.select_atoms(backbone)
    
    mobile = mda.Universe(pdb)
    align.AlignTraj(mobile, ref, select=backbone, in_memory=True).run()
    r = rms.rmsd(mobile.select_atoms(backbone).positions, ref_atoms.positions, superposition=True)
    g = mobile.select_atoms(backbone).radius_of_gyration()
    return r, g

########### Calculate gradient, CVs distance ##############
def calculate_gradients_distance(RMSD, Rg, Rg_end, Energy, t, txt_file_path):
    grad_distance_t = {
        'step': t,
    }

    with open(txt_file_path, 'r') as f:
        lines = f.readlines()

    for line in lines[1:]:
        data = line.strip().split()
        StructureID = data[0]
        Opt_RMSD = float(data[3])
        Opt_Rg = float(data[4])
        Opt_Energy = float(data[5])
        
        delta_RMSD = Opt_RMSD - RMSD
        delta_Rg = Opt_Rg - Rg
        delta_Energy = Opt_Energy - Energy

        distance_to_start = np.sqrt(delta_Rg**2)
        distance_to_end = np.sqrt((Opt_Rg-Rg_end)**2)
        distance_start_end = np.sqrt((Rg - Rg_end)**2)
        
        rmsd_grad = delta_Energy / delta_RMSD if delta_RMSD != 0 else np.nan
        rg_grad = delta_Energy / delta_Rg if delta_Rg != 0 else np.nan
        denominator = np.sqrt(delta_RMSD**2 + delta_Rg**2)
        CVs_grad = delta_Energy / denominator if denominator != 0 else np.nan
        
        grad_distance_t[StructureID] = {
            'CVs_start': [RMSD, Rg],
            'CVs_opt': [Opt_RMSD, Opt_Rg],
            'Opt_Energy': Opt_Energy,
            'rmsd_grad': rmsd_grad,
            'rg_grad': rg_grad,
            'CVs_grad': CVs_grad,
            'distance_to_start': distance_to_start,
            'distance_to_end': distance_to_end,
            'distance_start_end': distance_start_end
        }
    
    return grad_distance_t

def find_min_cvs_grad(grad_distance_t, distance_input_end, stepsize, initial_relax=0.0, relax_step=0.1, max_relax=10):

    relax_cof = initial_relax
    candidates = {}
    while relax_cof <= max_relax:
        for struct_id, data in grad_distance_t.items():
            if struct_id == "step":
                continue

            if data["distance_to_end"] < distance_input_end + relax_cof + 1e-9:
                candidates[struct_id] = {
                    "CVs_opt": data["CVs_opt"],
                    "CVs_grad": data["CVs_grad"],
                    "Opt_Energy": data["Opt_Energy"],
                    "distance_to_end": data["distance_to_end"]
                }
        if candidates:
            break
        relax_cof += relax_step
    
    if candidates is None:
        for struct_id, data in grad_distance_t.items():
            if struct_id == "step":
                continue
            candidates[struct_id] = {
                "CVs_opt": data["CVs_opt"],
                "CVs_grad": data["CVs_grad"],
                "Opt_Energy": data["Opt_Energy"],
                "distance_to_end": data["distance_to_end"]
            }

    min_id = min(candidates.keys(), key=lambda x: candidates[x]["CVs_grad"])
    return {
        "best_direction": min_id,
        "CVs": candidates[min_id]["CVs_opt"], 
        "Energy":candidates[min_id]["Opt_Energy"],
        "distance_to_end": candidates[min_id]["distance_to_end"],
        "CVs_grad":candidates[min_id]["CVs_grad"],
        "dt":stepsize
    }
