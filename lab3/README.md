# CSE5449 Lab 3 - Model Memory Overhead

Environment related instructions:

```bash

# to create a conda environment
source create-env.sh

# to activate the environment created after running the above once in root directory
source activate-env.sh

```

Commands to monitor GPU metrics:

```bash

watch -n .1 nvidia-smi


# to monitor power, utilization and memory, optional pipe output to file instead of STDOUT
nvidia-smi dmon -s pum [-f filename]

# monitor above as well as 
# 5 - tensor activity
# 11, 12, 13 - FP64, FP32, FP16 activity
nvidia-smi dmon -s pum --gpm-metrics 5,11,12,13

```

## EleutherAI/Pythia-410m 


```bash

# inference memory

python transformer_mem.py --num-layers=24 --sequence-length=2048 --num-attention-heads=16 --hidden-size=1024 --zero-stage=0 --pipeline-parallel-size=1 --tensor-parallel-size=1 --num-gpus=1 --params=410000000 --fp32-model --infer


# training memory

python transformer_mem.py --num-layers=24 --sequence-length=2048 --num-attention-heads=16 --hidden-size=1024 --zero-stage=0 --pipeline-parallel-size=1 --tensor-parallel-size=1 --num-gpus=1 --params=410000000


# inference

python infer.py --model_type=pythia --model_name_or_path=EleutherAI/pythia-410m

python infer.py --model_type=pythia --model_name_or_path=EleutherAI/pythia-410m --fp16


# training

python train.py \
    --model_name_or_path EleutherAI/pythia-410m \
    --dataset_name enwik8 \
    --dataset_config_name enwik8 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --do_train \
    --do_eval \
    --output_dir /tmp/train

python train.py \
    --model_name_or_path EleutherAI/pythia-410m \
    --dataset_name enwik8 \
    --dataset_config_name enwik8 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --do_train \
    --do_eval \
    --fp16 \
    --output_dir /tmp/train

```

## EleutherAI/Pythia-160m


```bash

# inference
python transformer_mem.py --num-layers=12 --sequence-length=2048 --num-attention-heads=12 --hidden-size=768 --zero-stage=0 --pipeline-parallel-size=1 --tensor-parallel-size=1 --num-gpus=1 --params=160000000 --fp32-model --infer


# training

python transformer_mem.py --num-layers=12 --sequence-length=2048 --num-attention-heads=12 --hidden-size=768 --zero-stage=0 --pipeline-parallel-size=1 --tensor-parallel-size=1 --num-gpus=1 --params=160000000


# inference

python infer.py --model_type=pythia --model_name_or_path=EleutherAI/pythia-160m


python infer.py --model_type=pythia --model_name_or_path=EleutherAI/pythia-160m --fp16

# training

python train.py \
    --model_name_or_path EleutherAI/pythia-160m \
    --dataset_name enwik8 \
    --dataset_config_name enwik8 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --do_train \
    --do_eval \
    --output_dir /tmp/train

python train.py \
    --model_name_or_path EleutherAI/pythia-160m \
    --dataset_name enwik8 \
    --dataset_config_name enwik8 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --do_train \
    --do_eval \
    --fp16 \
    --output_dir /tmp/train

```