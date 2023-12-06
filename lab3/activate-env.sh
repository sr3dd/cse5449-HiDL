module purge
module load gcc-compatibility/10.3.0 cuda/11.6.1
#source /fs/ess/PAS2581/owens/load_mv2
export PYTHONNOUSERSITE=true

source $PWD/environment/miniconda/bin/activate
conda activate pytorch