#!/bin/bash

minim2_dir=$1
output_file="${minim2_dir}/2jof_output.txt"
pdb_source_dir="${minim2_dir}/2jof_pdb_opt"
em1_dir="./2jof_pdb_em"
noem1_dir="./2jof_pdb_noem"
em1_list_file="2jof_pdb_em.txt"
noem1_list_file="2jof_pdb_noem.txt"

mkdir -p "$em1_dir"
mkdir -p "$noem1_dir"

#> "$em1_list_file"

total_count=0
em1_count=0
noem1_count=0

process_file() {
    local pdb_number="$1"
    local energy="$2"
    
    local source_pdb="${pdb_source_dir}/2jof_em_${pdb_number}.pdb"
    
    if [ ! -f "$source_pdb" ]; then
        echo "Warnings: File ${source_pdb} doesn't exist"
        return 1
    fi
    
    ((total_count++))
    
    if awk -v e="$energy" 'BEGIN {exit !(e < 0)}'; then
        cp "$source_pdb" "$em1_dir/"
        echo "${pdb_number} ${energy}" >> "$em1_list_file"
        ((em1_count++))
        echo "Copy ${source_pdb} to ${em1_dir} (Energy=${energy})"
    else
        cp "$source_pdb" "$noem1_dir/"
	echo "${pdb_number} ${energy}" >> "$noem1_list_file"
        ((noem1_count++))
        echo "Copy ${source_pdb} to ${noem1_dir} (Energy=${energy})"
    fi
}

# Main function
while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    [[ "$line" == \#* ]] && continue
    
    pdb_number=$(echo "$line" | awk '{print $1}')
    energy=$(echo "$line" | awk '{print $2}')
    
    process_file "$pdb_number" "$energy"
done < "$output_file"

# Output
echo "===================================="
echo "Process completed"
echo "Total number of files processed: ${total_count}"
echo "Number of files with potential energy < 0: ${em1_count} (Copied ${em1_dir})"
echo "Number of files with potential energy > 0: ${noem1_count} (Copied ${noem1_dir})"
echo "The record of potential energy < 0 has been saved to: ${em1_list_file}"
echo "===================================="
