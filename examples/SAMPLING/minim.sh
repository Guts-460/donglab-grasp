#!/bin/bash
module load GROMACS/2018.8-cpu-new

# 解析命令行参数
while getopts "i:c:o:x:s:l:n:" opt; do
  case $opt in
    i) pdb_dir_input="$OPTARG" ;;
    c) itp_dir_input="$OPTARG" ;;
    o) output_pdb_dir="$OPTARG" ;;
    x) output_txt_dir="$OPTARG" ;;
    s) sample_step="$OPTARG" ;;
    l) log_dir="$OPTARG" ;;
    n) num_directions="$OPTARG" ;;
    \?) echo "Invalid option -$OPTARG" >&2; exit 1 ;;
  esac
done


# for t in {1..8}; do
for ((t=1; t<=num_directions; t++)); do
    pdb_path="${pdb_dir_input}/sidechain_${t}.pdb"     
    # itp="sidechain_${t}.itp"    
    # cp "${pdb_dir_input}/${pdb}" .
    # cp "${itp_dir_input}/${itp}" .
    # mv "$itp" dihedral.itp
    
    gmx_d pdb2gmx -f "$pdb_path" -o ./${log_dir}/em-st-i-box.gro -water none -ignh <<EOF
8
EOF
    mv topol.top ./${log_dir}
    mv posre.itp ./${log_dir}
    # tac topol.top | sed '8i INSERTITP' | tac > temp.top && mv temp.top topol.top
    # sed -i 's/INSERTITP/#include "dihedral.itp"\n/' topol.top

    for i in {1..2}; do
        gmx_d grompp -f ./common/mdp/minim-steep-nopbc.mdp -c ./${log_dir}/em-st-i-box.gro -p ./${log_dir}/topol.top -po ./${log_dir}/em-st -pp ./${log_dir}/em-st -o ./${log_dir}/em-st.tpr -maxwarn 1
        gmx_d mdrun -v -deffnm ./${log_dir}/em-st -nt 2 -pin on -pinoffset 0 -pinstride 1
        mv ./${log_dir}/em-st.gro ./${log_dir}/i_em.gro
        rm ./${log_dir}/em-st*
        mv ./${log_dir}/i_em.gro ./${log_dir}/em-st-i-box.gro
    done
    mv ./${log_dir}/em-st-i-box.gro ./${log_dir}/em-st-i.gro
    
    for i in {1..2}; do
        gmx_d grompp -f ./common/mdp/minim-cg-nopbc.mdp -c ./${log_dir}/em-st-i.gro -p ./${log_dir}/topol.top -po ./${log_dir}/em-st -pp ./${log_dir}/em-st -o ./${log_dir}/em-st.tpr -maxwarn 1
        gmx_d mdrun -v -deffnm ./${log_dir}/em-st -nt 2 -pin on -pinoffset 0 -pinstride 1
        mv ./${log_dir}/em-st.gro ./${log_dir}/i_em.gro
        mv ./${log_dir}/em-st.log ./${log_dir}/i_em.log
        rm ./${log_dir}/em-st*
        mv ./${log_dir}/i_em.gro ./${log_dir}/em-st-i.gro
        mv ./${log_dir}/i_em.log ./${log_dir}/em-st-i.log
    done

    gmx_d editconf -f ./${log_dir}/em-st-i.gro -o "./${log_dir}/opt_${t}.pdb"
    mv "./${log_dir}/opt_${t}.pdb" "$output_pdb_dir"
  
    if [ -f "./${log_dir}/em-st-i.log" ]; then
        grep 'Potential Energy' ./${log_dir}/em-st-i.log | awk -v i_val=" " '{print i_val $4}' >> ./${log_dir}/ec_op1.txt

        et=$(tail -n 1 ./${log_dir}/ec_op1.txt | awk '{print $1}')

        echo "$t $et" >> "output_${sample_step}.txt"

        mv "./${log_dir}/em-st-i.log" "./${log_dir}/opt_${t}.log"
        mv "./${log_dir}/opt_${t}.log" "$output_pdb_dir"
    fi

    rm ./${log_dir}/em*
    rm ./${log_dir}/topol.top
    rm ./${log_dir}/ec_op1.txt
    # rm ./${log_dir}/dihedral.itp
    rm ./${log_dir}/posre.itp
    # rm ./${log_dir}/"$pdb"
done

mv "output_${sample_step}.txt" "$output_txt_dir"
