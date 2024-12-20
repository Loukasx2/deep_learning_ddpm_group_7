#!/bin/sh
### ------------- specify queue name ---------------- 
#BSUB -q gpua100

### ------------- specify gpu request---------------- 
#BSUB -gpu "num=1:mode=exclusive_process"

### ------------- specify job name ---------------- 
#BSUB -J cifar_new

### ------------- specify number of cores ---------------- 
#BSUB -n 4 
#BSUB -R "span[hosts=1]"

### ------------- specify CPU memory requirements ---------------- 
#BSUB -R "rusage[mem=30GB]"

#BSUB -W 72:00 
#BSUB -o output2/OUTPUT_FILE%J.out 
#BSUB -e output2/OUTPUT_FILE%J.err

source "/dtu/blackhole/1e/203934/ddpm/deep_learning_ddpm_group_7/.venv_a100/bin/activate"
# cd /dtu/blackhole/1e/203934/ddpm/deep_learning_ddpm_group_7/
python -m tools.train_ddpm --config config/training.yaml
# bash /dtu/blackhole/1e/203934/ddpm/deep_learning_ddpm_group_7/tools/sampling.sh
# python -m tools.sample_ddpm --config config/training.yaml