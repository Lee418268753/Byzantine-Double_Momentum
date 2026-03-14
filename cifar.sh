#!/bin/bash
export MKL_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

source ./parallelize.sh
set -u
echo "Starting at $(date)"


GPU_IDS=(0 1 2 3)

NUM_GPUS=${#GPU_IDS[@]}

gpu_idx=0

cmds=""
njobs=80 
datasets="cifar10"
b="64"
e="100"
n="20"
f="8"

for lr in 0.5 0.05 0.005 
do
for model in dasha #ef21 sgd diana #sgd diana marina
do
for seed in 123 124 125
do
for attack in SF LF #IPM ALIE SF LF NA 
do
for agg in rfa cm cwtm #cp tm
do
    if [ "$model" = "top_sgd" ] || [ "$model" = "ef" ]; then
        compression="contractive_compressorcnn"
    else
        compression="random_sparsificationcnn"
    fi
    current_gpu=${GPU_IDS[$gpu_idx]}
    
    cmds="$cmds ;CUDA_VISIBLE_DEVICES=$current_gpu python run_a9a.py --agg $agg --attack $attack --datasets $datasets --model $model --lr $lr --batch-size $b --epochs $e -n $n -f $f --use-cuda --eval-every 1 --noniid --seed $seed --nnm --compression $compression "
    
    gpu_idx=$(( (gpu_idx + 1) % NUM_GPUS ))
done
done
done
done
done

set +u
f_ParallelExec $njobs "$cmds"


echo "Done at $(date)"
