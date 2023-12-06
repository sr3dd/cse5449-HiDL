mkdir $PWD/environment/
cd $PWD/environment/
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $PWD/miniconda
source $PWD/miniconda/bin/activate
cd ..
conda create -n pytorch python=3.9
conda activate pytorch
pip install -r requirements.txt
