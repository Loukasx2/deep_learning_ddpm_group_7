#!/bin/bash

# Number of iterations to run
NUM_ITERATIONS=36

# Activate the virtual environment (if needed)
# source /path/to/your/venv/bin/activate

for ((i=1; i<=NUM_ITERATIONS; i++))
do
    echo "Iteration $i of $NUM_ITERATIONS"
    python -m tools.sample_ddpm

    if [ $? -ne 0 ]; then
        echo "Error encountered in iteration $i. Exiting..."
        exit 1
    fi

done

# Deactivate the virtual environment (if activated)
# deactivate
