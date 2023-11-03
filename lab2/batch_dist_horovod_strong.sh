#!/bin/bash

export LANGUAGE=en_US.UTF-8
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8

export LD_PRELOAD=""
MV2_SUPPORT_DL=1 MV2_HYBRID_BINDING_POLICY=spread MV2_CPU_BINDING_POLICY=hybrid MV2_USE_ALIGNED_ALLOC=1 MV2_USE_CUDA=1 MV2_HOMOGENEOUS_CLUSTER=1 MV2_ENABLE_TOPO_AWARE_COLLECTIVES=0 MV2_CUDA_BLOCK_SIZE=8388608 python dist_horovod_resnet.py --batch-size 1024 --strong-scale
