export LANGUAGE=en_US.UTF-8
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8

export MV2_DEBUG_SHOW_BACKTRACE=2
export MV2_SUPPORT_DL=1
export MV2_HYBRID_BINDING_POLICY=spread
export MV2_CPU_BINDING_POLICY=hybrid
export MV2_USE_ALIGNED_ALLOC=1
export MV2_USE_CUDA=1 
export MV2_HOMOGENEOUS_CLUSTER=1 
export MV2_ENABLE_TOPO_AWARE_COLLECTIVES=0
export LD_PRELOAD=""

srun -n 2 --export=ALL,MV2_USE_CUDA=1,MV2_SUPPORT_DL=1 python pytorch_ddp_implemented.py 2> /dev/null