#!/bin/bash
#SBATCH -JHF_qz
#SBATCH -Ahive-cs207
#SBATCH -N50 --ntasks=50 --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH -t4:00:00
#SBATCH -phive
#SBATCH -oreport-%j.out
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=awallace43@gatech.edu

cd $HOME/data/SAPT_D4
module load anaconda3/2022.05.0.1
module load openmpi/4.1.4
conda activate /storage/hive/project/chem-sherrill/sherrill-software/.conda/envs/qcn

echo "starting job..."
mpiexec -n 50 python3 main.py
echo "finished job!"

