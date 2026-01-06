import os
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import rms, align
from tqdm import tqdm
import time

def calculate_rmsd_rg(pdb_dir, ref_pdb, output_file):
    print(f"Load reference structure:{ref_pdb}")
    ref = mda.Universe(ref_pdb)
    
    atom_selection = "protein and (name N or name CA or name C or name O or name OT1)"
    ref_atoms = ref.select_atoms(atom_selection)
    
    with open(output_file, 'w') as f:
        f.write("code\tRMSD\tRg\n")
    
    pdb_files = sorted([f for f in os.listdir(pdb_dir) if f.endswith('.pdb')], 
                      key=lambda x: int(x.split('_')[-1].split('.')[0]))
    print(f"Found  {len(pdb_files)} PDB files that need to be processed")

    batch_size = 2000
    batches = [pdb_files[i:i + batch_size] for i in range(0, len(pdb_files), batch_size)]

    for batch_num, batch in enumerate(batches, 1):
        print(f"\nbatch processing {batch_num}/{len(batches)} (Total {len(batch)} files)")
        batch_results = []
        
        for filename in tqdm(batch):
            try:
                code = filename.split('_')[-1].split('.')[0]

                mobile = mda.Universe(os.path.join(pdb_dir, filename))
                aligner = align.AlignTraj(mobile, ref, select=atom_selection, in_memory=True)
                aligner.run()
    
                rmsd = rms.rmsd(mobile.select_atoms(atom_selection).positions, 
                               ref_atoms.positions, 
                               superposition=True)

                selected_atoms = mobile.select_atoms(atom_selection)
                rg = selected_atoms.radius_of_gyration()
                
                batch_results.append(f"{code}\t{rmsd:.4f}\t{rg:.4f}\n")
            except Exception as e:
                print(f"\nProcess {filename} Error: {str(e)}")
                batch_results.append(f"{code}\tERROR\tERROR\n")

        with open(output_file, 'a') as f:
            f.writelines(batch_results)
        
        time.sleep(1)
    
    print(f"\nAll processing completed!")
    print(f"The RMSD and Rg results have been saved to: {output_file}")

if __name__ == "__main__":
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        current_dir = os.getcwd()
        print(f"Detected an interactive environment, using the current working directory: {current dir}")
    
    pdb_dir = os.path.join(current_dir, "2jof_pdb_em")
    ref_pdb = os.path.join(current_dir, "./2jof_pdb_em/2jof_em_032000.pdb") # Reference conformation
    output_file = os.path.join(current_dir, "2jof_rmsd_rg_em_032000.txt")
    
    if not os.path.exists(pdb_dir):
        raise FileNotFoundError(f"PDB directory does not exist: {pdc_dir}")
    if not os.path.exists(ref_pdb):
        raise FileNotFoundError(f"Reference PDB does not exist: {ref_pdb}")
    
    calculate_rmsd_rg(pdb_dir, ref_pdb, output_file)