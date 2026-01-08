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

An application can be found at branch "Examples"

## Theory <br>
Please read our manuscript titled as "..." <br>

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
SCWRL4: https://dunbrack.fccc.edu/lab/scwrl <br>
MCS (Recommend, but unnecessary) <br>

## 1 Data set for VAE
Sparse conformational collection refers to a discretized representation of a protein’s conformational ensemble, and multiple construction strategies exist. For mini-proteins (10–30 residues), a discrete conformational space can typically be obtained by randomly sampling backbone dihedral angles followed by structural refinement. For mid-proteins in this work, we used a annealing simulation to process random backbones and avoided lots of Kinetically unreachable conformations. In the future, for larger proteins, including middle and high weight proteins, we will test the loop-helix-loop unit combinatorial sampling algorithm (LUCS)1, previously shown to be capable of generating static proteins that differ in the local geometry of user-defined protein segments, and AlphaFold tools2 to enhance the quality of data set.

