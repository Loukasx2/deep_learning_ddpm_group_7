#!/bin/sh
### ------------- specify queue name ---------------- 
#BSUB -q gpua100

### ------------- specify gpu request---------------- 
#BSUB -gpu "num=1:mode=exclusive_process"

### ------------- specify job name ---------------- 
#BSUB -J sample_20_layers

### ------------- specify number of cores ---------------- 
#BSUB -n 4 
#BSUB -R "span[hosts=1]"

### ------------- specify CPU memory requirements ---------------- 
#BSUB -R "rusage[mem=30GB]"

#BSUB -W 24:00 
#BSUB -o output/OUTPUT_FILE%J.out 
#BSUB -e output/OUTPUT_FILE%J.err

source /dtu/blackhole/1e/203934/ddpm/deep_learning_ddpm_group_7/.venv_a100/bin/activate
# python -m tools.train_ddpm --config config/train_a100.yaml
cd /dtu/blackhole/1e/203934/ddpm/deep_learning_ddpm_group_7/
# python -m tools.train_ddpm --config config/training.yaml
# python -m tools.sample_ddpm --config config/train_a100.yaml
bash /dtu/blackhole/1e/203934/ddpm/deep_learning_ddpm_group_7/tools/sampling.sh