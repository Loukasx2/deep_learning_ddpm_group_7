#!/bin/sh
### ------------- specify queue name ---------------- 
#BSUB -q gpuv100

### ------------- specify gpu request---------------- 
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "select[gpu32gb]"

### ------------- specify job name ---------------- 
#BSUB -J testjob_ddpm_cifar2

### ------------- specify number of cores ---------------- 
#BSUB -n 4 
#BSUB -R "span[hosts=1]"

### ------------- specify CPU memory requirements ---------------- 
#BSUB -R "rusage[mem=30GB]"

#BSUB -W 12:00 
#BSUB -o output/OUTPUT_FILE%J.out 
#BSUB -e output/OUTPUT_FILE%J.err

source "/zhome/d8/f/203934/ddpm/deep_learning_ddpm_group_7/.venv/bin/activate"
python -m tools.train_ddpm
python -m tools.sample_ddpm