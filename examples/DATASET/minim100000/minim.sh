#!/bin/bash
module load GROMACS/2018.8-cpu-new

# 创建必要的目录
mkdir -p $1


for t in {65001..100000}; do
    # 格式化索引为6位数，前面补零
    a=$(printf "%06d" $t)
    
    # 处理当前任务
    pdb=2jof_"$a".pdb      #输入待优化pdb
    #itp=2jof_"$a".itp  #输入待限制的二面角

    pdb_dir_input=mcs/2jof_dir
    #itp_dir_input=2jof_itp_dir

    cp ../.././$pdb_dir_input/$pdb .  
    #cp .././$itp_dir_input/$itp .

    #mv $itp dihedral.itp
    gmx_d pdb2gmx -f $pdb -o em-st-i-box.gro -water none -ignh <<EOF
8
EOF

    tac topol.top | sed '8i INSERTITP' | tac > temp.top && mv temp.top topol.top
    # sed -i 's/INSERTITP/#include "dihedral.itp"\n/' topol.top

    for i in {1..2}; do
        gmx_d grompp -f minim-steep-nopbc.mdp -c em-st-i-box.gro -p topol.top -po em-st -pp em-st -o em-st.tpr -maxwarn 1
        gmx_d mdrun -v -deffnm em-st -nt 2 -pin on -pinoffset 4 -pinstride 1
        mv em-st.gro i_em.gro
        rm em-st*
        mv i_em.gro em-st-i-box.gro
    done

    # gmx_d editconf -f em-st-i-box.gro -box 8 8 8 -o em-st-i.gro -c
    mv em-st-i-box.gro em-st-i.gro
    
    for i in {1..2}; do
        gmx_d grompp -f minim-cg-nopbc.mdp -c em-st-i.gro -p topol.top -po em-st -pp em-st -o em-st.tpr -maxwarn 1
        gmx_d mdrun -v -deffnm em-st -nt 2 -pin on -pinoffset 4 -pinstride 1
        mv em-st.gro i_em.gro
        mv em-st.log i_em.log
        rm em-st*
        mv i_em.gro em-st-i.gro
        mv i_em.log em-st-i.log
    done

    gmx_d editconf -f em-st-i.gro -o 2jof_em_"$a".pdb
    mv 2jof_em_"$a".pdb ./$1

    grep 'Potential Energy' em-st-i.log | awk -v i_val=" " '{print i_val $4}' >> ec_op1.txt

    et=$(tail -n 1 ec_op1.txt | awk '{print $1}')

    # 将能量汇总（pdb序号，势能）
    echo "$a $et" >> 2jof_output.txt

    mv em-st-i.log 2jof_em_"$a".log
    mv 2jof_em_"$a".log ./$1

    rm em*
    rm topol.top
    rm ec_op1.txt
    #rm dihedral.itp
    rm posre.itp
    rm $pdb
done
