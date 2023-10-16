module purge
module load gcc-compatibility/10.3.0 cuda/11.6.1
source /fs/ess/PAS2581/owens/load_mv2
source ~/owens/lab1/miniconda3/bin/activate
conda activate mpi4py
export PYTHONNOUSERSITE=true
