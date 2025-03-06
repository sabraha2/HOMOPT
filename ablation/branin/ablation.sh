#!/bin/bash
# This bash script runs the ablation study for the Branin function.

# Ablation study parameters:
# - inner_method: (bayes, random, smac, tpe)
# - k: fraction of data used for the inner method (0.1 to 1)
# - iterations: number of iterations to run the optimization (1 to 10)
# - jitter_strength: strength of the jitter (various scales)
# - warm_up: number of initial warm-up iterations (10 to 100)
# - n_splines: number of splines for GAM (varied)
# - lambda: regularization strength for GAM (varied)

# The results of the ablation study are saved in the results folder.

# Ensure results directory exists
mkdir -p results

for inner_method in bayes random smac tpe; do
    for k in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1; do
        for iterations in 1 2 3 4 5 6 7 8 9 10; do
            for jitter_strength in 5 0.5 0.05 0.005 0.0005 0.00005; do
                for warm_up in 10 20 30 40 50 60 70 80 90 100; do
                    for n_splines in 10 15 20 25 30; do
                        for lambda in 0.1 0.01 0.001 0.0001 0.00001; do
                            result_path="results/$inner_method/k_$k/iterations_$iterations/jitter_strength_$jitter_strength/warm_up_$warm_up/splines_$n_splines/lambda_$lambda"
                            if [ -d "$result_path" ]; then
                                echo "Results already exist for $inner_method k=$k iter=$iterations jitter=$jitter_strength warmup=$warm_up splines=$n_splines lambda=$lambda, skipping..."
                                continue
                            fi
                            echo "Running: $inner_method k=$k iter=$iterations jitter=$jitter_strength warmup=$warm_up splines=$n_splines lambda=$lambda"
                            python3 driver.py --inner_method $inner_method --k $k --max_iter $iterations --jitter_strength $jitter_strength --warm_up $warm_up --n_splines $n_splines --lambda $lambda
                            if [ -f results.json ]; then
                                mkdir -p $result_path
                                mv results.json $result_path
                            fi
                        done
                    done
                done
            done
        done
    done
done
