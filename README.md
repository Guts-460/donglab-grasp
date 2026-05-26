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

## Code location <br>
The training and testing code for the potential well model is located in the **ADWP** branch (https://github.com/Guts-460/donglab-grasp/tree/ADWP) <br>
Find data available and examples for testing, please refer to branch **Trp-cage** (https://github.com/Guts-460/donglab-grasp/tree/Trp-cage) <br>
All other training and sampling code are as same as Trp-cage's <br>

## Theory <br>
<center>
<img width="1200" height="600" alt="image" src="https://github.com/user-attachments/assets/54dd12cc-d57b-4bfd-a6f4-ddad900db5f8" /> <br>
<center>
Fig. 1. Workflow of the DA2-GRASP algorithm. <br>

## Environment Deployment <br>
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
