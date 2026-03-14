#! /bin/bash
export MKL_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

source ./parallelize.sh
set -u
echo "Starting at $(date)"
CUDA_VISIBLE_DEVICES="0"
datasets="a9a"
njobs=20
cmds=""
testbatchsize="32561"
b="1"
e="40"
n="20"
f="8"
gpu_index=0
for lr in 0.5 0.05 0.005
do
for model in dasha #sgd diana marina
do
for seed in 123
do
for attack in IPM ALIE SF LF NA 
do
for agg in rfa cm cwtm  #cp tm
do

    if [ "$model" = "top_sgd1" ] || [ "$model" = "ef21" ]; then
        compression="contractive_compressorlr"
    else
        compression="random_sparsificationlr"
    fi
    if [ "$agg" = "avg" ]; then
        cmds="$cmds ;CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python run_lr.py --agg $agg --attack $attack --datasets $datasets --model $model --lr $lr --batch-size $b --epochs $e -n $n -f $f --test-batch-size $testbatchsize --use-cuda --eval-every 1 --seed $seed --compression $compression "

    else
        cmds="$cmds ;CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python run_lr.py --agg $agg --attack $attack --datasets $datasets --model $model --lr $lr --batch-size $b --epochs $e -n $n -f $f --test-batch-size $testbatchsize --use-cuda --eval-every 1 --seed $seed --nnm --compression $compression "

    fi
    done
done
done
done
done
set +u
f_ParallelExec $njobs "$cmds"


echo "Done at $(date)"

