# donglab-grasp
## A Deep Generativate Model Sampling Protein Favorable Folding Pathway <br>
**Author**: Yanbing Wen, & Hao Dong* <br>
<br>
**Institution**:
State Key Laboratory of Analytical Chemistry for Life Science, <br>
Kuang Yaming Honors School, Chemistry and Biomedicine Innovation Center (ChemBIC), <br>
ChemBioMed Interdisciplinary Research Center at Nanjing University, & Institute for Brain Sciences, Nanjing University, Nanjing 210023, China. <br>
<br>
We are still updating this repository. <br>
All code will be shared once manuscript is accepted.<br>

An application can be found at branch "Examples"  <br>

## Theory <br>
Please read our manuscript titled as "..." <br>
<center>
<img width="1200" height="600" alt="image" src="https://github.com/user-attachments/assets/54dd12cc-d57b-4bfd-a6f4-ddad900db5f8" /> <br>
<center>
Fig. 1. Workflow of the DA2-GRASP algorithm. <br>

## 0 Environment Deployment <br>
cudatoolkit = 11.8 <br>
torch = 2.0.0 <br>
torchvision = 0.15.1 <br>
torchaudio = 2.0.1 <br>
prody = 2.6.1 <br>
scikit-learn = 1.7.1 <br>
mdanalysis = 2.9 <br>
numpy = 1.25 <br>
GROMACS (2018.8, cpu) : https://manual.gromacs.org/2018.8/download.html <br>
SCWRL4[1]: https://dunbrack.fccc.edu/lab/scwrl <br>
MCS (Recommend, but unnecessary) <br>

## 1 Data set for VAE
Sparse conformational collection refers to a discretized representation of a protein’s conformational ensemble, and multiple construction strategies exist. For mini-proteins (10–30 residues), a discrete conformational space can typically be obtained by randomly sampling backbone dihedral angles followed by structural refinement. For mid-proteins in this work, we used a annealing simulation to process random backbones and avoided lots of Kinetically unreachable conformations. In the future, for larger proteins, including middle and high weight proteins, we will test the loop-helix-loop unit combinatorial sampling algorithm (LUCS)[2], previously shown to be capable of generating static proteins that differ in the local geometry of user-defined protein segments, and AlphaFold tools2 to enhance the quality of data set. <br>

### 1.1 Randomly conformations
We removed the Metropolis criterion3 from the Monte Carlo simulation (MCS) protocol to rapidly sample backbone dihedral angles and generate unbiased random protein backbones.（**Site-packages/mcsoftware-corr-noP.tar**） <br>
As an example, like trap-cage (or chignolin), you need prepare a file (.angs) describing the original distribution of dihedral angles, no matter its state as below: <br>
<center>
<img width="600" height="421" alt="image" src="https://github.com/user-attachments/assets/a2cf4a52-b9dd-4534-9892-3c7e2e3ae778" /> <br>
<center>

Then, run commands <br>
mkdir 2jof_dir <nr>
nohup ./mcs -I 2jof.angs -S 100000 -N 1 -K 1 -A 2 -F 1 -R 1 -O 2jof -X 2jof_dir > 2jof_dir.log 2>&1 & <br>

On a single-core CPU, you will obtain 100,000 random backbones (mcs/2jof_dir) —each with the same chain length as Trap-cage—within 20 minutes (**Fig. 2**).  <br>
<center>
<img width="800" height="600" alt="image" src="https://github.com/user-attachments/assets/bcc6db0d-e903-4821-ac17-a6d6e662253e" /> <br>
 <center>
Fig. 2 Random backbones <br>

### 1.2 Optimization
Subsequently, the random conformations require a few hundred steps of conformational optimization to eliminate unphysical features such as incorrect bond lengths, bond angles, and dihedral angles (**cd A-DATASET**). <br>
mkdir minim30000 minim65000 minim100000 <br>
 <br>
cd minim30000 <br>
nohup ./minim.sh 2jof_pdb_opt > 2jof_opt.log 2>&1 & <br>
 <br>
cd minim65000 <br>
nohup ./minim.sh 2jof_pdb_opt > 2jof_opt.log 2>&1 & <br>
 <br>
cd minim100000 <br>
nohup ./minim.sh 2jof_pdb_opt > 2jof_opt.log 2>&1 & <br>
 <br>

For any optimization process, you will receive the following files: <br>
./2jof_pdb_opt contains all optimized conformations, <br>
2jof_opt.log contains optimization logs <br>
2jof_output.txt contains information including code & energy <br>

### 1.3 Extract pdbs
We prepared a bash script (**A-DATASET/extract_pdb.sh**) to extract structure optimized from 2jof_pdb_opt at given energy cutoff, like lower than 0 kJ/mol. You need define work files as below: <br>
<center>
<img width="400" height="260" alt="image" src="https://github.com/user-attachments/assets/fc2e52ed-c02f-49b8-9859-6e15add4b905" /> <br>
 <center>
./minim2_dir is the contents like minim30000, minim65000 or minim100000 <br>
 <br>
./em1_dir save structures with energy < cutoff. <br>
2jof_pdb_em.txt save energies < cutoff. <br>
./noem1_dir save structures with energy > cutoff. <br>

Then, cd A-DATASET and run command: <br>
./extract_pdb.sh minim30000 <br>
./extract_pdb.sh minim65000 <br>
./extract_pdb.sh minim100000 <br>
All structures with energy < cutoff will be saved to 2jof_pdb_em, associated energy saved to 2jof_pdb_em.txt. <br>
 
### 1.4 Extract CVs
We also prepared a python script to extract collective variables (CVs), associating any dynamic motion you want to study. In our work, we trained conformational transition with condition of ΔRMSD & ΔRg. Define the path and reference structure in the script (**A-DATASET/rg_rmsd_cal.py**) as below: <br>
<center>
<img width="600" height="67" alt="image" src="https://github.com/user-attachments/assets/0c1ce692-ba6d-4e26-9ead-39b761acac01" />  <br>
 <center>
./pdb_dir include structures saved with energy < cutoff, ref_pdb is the reference structure for RMSD-CA, and randomly selected from ./pdb_dir. <br>
Then, run command: nohup python rg_rmsd_cal.py ./rg_rmsd_cal.log 2>&1 & <br> 
All aligned structures will be saved in 2jof_aligned, all rmsd & rg will be saved in 2jof_rmsd_rg_em_032000.txt. <br>
<br>
**It should be noted that the dataset construction strategy is not unique; the protocol we provide serves only as a reference. Acquiring as many conformations as possible, along with their corresponding energies, will enable the model to capture richer conformational transition features.** <br>

## 2 Train DA2-GRASP
Once we have completed the construction of the data set (**2jof_aligned & 2jof_rmsd_rg_em_032000.txt**), we can start training DA2-GRASP, a thermodynamically favorable path sampling framework that combines deep generative models, data-driven approaches, and physical gradients. Change work content to **./B-TRAINING**.  <br>
### 2.1 VAE
We use the coordinates of the protein backbone atoms (C, N, CA, O) as both the input and output of the variational autoencoder (VAE, **train_vae.py**). You can adapt the model to different systems by tuning the hyperparameters shown in the figure below.  <br>
<center>
<img width="600" height="372" alt="image" src="https://github.com/user-attachments/assets/80f8bc31-6494-4701-924a-9ba27deda177" />  <br>
 <center>
Run command: nohup python train_vae.py > train_vae.log 2>&1 & <br>
All loss values are saved in loss/loss_vae.txt, and the network parameters are stored in models/vae.pth and models/vae-scaler.pth. <br>

### 2.2 DataSet for Latent Conformation Transition
To enable conformational transitions in the latent space (**Fig. 1a**), we trained a mapping model that takes (ΔCVij, hi) as input and predicts hj as output—thereby learning to transform conformation i into conformation j under the condition specified by ΔCVij. We provide a Python script (**data-mapping.py or data-mapping.ipynb**) that (1) extracts latent features (h) for each conformation and (2) constructs the training dataset comprising tuples of (ΔCVij, hi, hj). <br>

### 2.3 Transformer-encoder
An attention mechanism is employed to assess how much the chosen (CVs) attend to structural features, thereby evaluating their relevance and usefulness. Accordingly, we adopt a Transformer-encoder architecture as the mapping module for conformational transitions (Train-mapping.py). This design offers two key advantages:  <br>
Ⅰ. Multi-head attention layers reduce reliance on any single CV by dynamically weighting their contributions; <br>
Ⅱ. Feed-forward neural network (FNN) layers actively drive the transformation from conformation (i) to conformation (j). <br>
 <br>
## 3 Sampling
Next, we are preparing to sample the folding path of the Trap cage. Firstly, organize the files in the following format： <br>
<center>
<img width="1200" height="726" alt="image" src="https://github.com/user-attachments/assets/b5fe9944-41c1-43e3-a71d-672aa6f6d271" />  <br>
<center>

The descriptions are shown in **Table 1** <br>
## File List

| File Name             | Description                                                                 |
|-----------------------|-----------------------------------------------------------------------------|
| `minim.sh`            | The conformational optimization module of DA2-GRASP, which adjusts bond lengths and angles to a reference state defined by the CHARMM27 force field, without inducing large global changes in the conformation. |
| `run.sh`              | Executes sampling commands, including protein folding and unfolding.         |
| `sampling_f2u.py`     | Samples protein unfolding.                                                   |
| `sampling_u2f.py`     | Samples protein folding.                                                     |
| `train_mapping.py`    | Implements a Transformer-encoder architecture for training the mapping module parameters. |
| `utils.py`            | Defines sampling functions, including CV value updates and gradient selection. |
| `pairwise-train-val.json` | Unified parameter dataset for the Transformer-encoder architecture.       |
| `minim-cg-nopbc.mdp`  | DA2-GRASP conjugate gradient optimization module: non-periodic, implicit solvent model. |
| `minim-steepest-nopbc.mdp` | DA2-GRASP steepest descent optimization module: non-periodic, implicit solvent model. |
| `2jof_mapping.pth`    | Model parameters for the mapping module.                                    |
| `2jof_vae.pth`        | VAE model parameters.                                                       |
| `vae-scaler.pkl`      | Min-max normalization coefficients for backbone atom coordinates.           |
| `2jof_Rg_small.pdb`   | A Trap-cage conformation with small Rg, used for sampling protein unfolding. |
| `2jof_Rg_large.pdb`   | A Trap-cage conformation with large Rg, used for sampling protein folding.   |
| `2jof_ref.pdb`        | Native state of Trap-cage, used for monitoring protein folding pathways; not involved in sampling. |
| `2jof_template.pdb`   | Backbone template of Trap-cage, used to replace the backbone coordinates generated by DA2-GRASP. |

Table 1 Functions for sampling protein folding.<br>


## Reference
1	Krivov, G. G., Shapovalov, M. V. & Dunbrack, R. L. Improved prediction of protein side-chain conformations with SCWRL4. Proteins 77, 778-795 (2009). https://doi.org:10.1002/prot.22488 <br>
2	Pan, X. J. et al. Expanding the space of protein geometries by computational design of de novo fold families. Science 369, 1132-+ (2020). https://doi.org:10.1126/science.abc0881 <br>
3	Abramson, J. et al. Accurate structure prediction of biomolecular interactions with AlphaFold 3. Nature 630 (2024). https://doi.org:10.1038/s41586-024-07487-w <br>
4	Lazaridis, T. & Karplus, M. Effective energy function for proteins in solution. Proteins 35, 133-152 (1999). https://doi.org:Doi 10.1002/(Sici)1097-0134(19990501)35:2<133::Aid-Prot1>3.0.Co;2-N <br>
