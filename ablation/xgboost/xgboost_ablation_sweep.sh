#!/usr/bin/env bash

timeout="30000"
max_tasks="100"
n_trials=5

queue="long@@cvrl"
pe="smp 1"

base="$(realpath $(pwd))"
results_dir="$base/xgboost_ablation"
mkdir $results_dir

for inner_method in bayes random smac tpe; do
    echo $inner_method
    
    for k in 0.2 0.4 0.6 0.8 1; do
        echo $k 
        
        for iterations in 3 6 9; do
            echo $iterations
            
            for jitter_strength in 5 5e-1 5e-2 5e-3 5e-4 5e-5; do
                echo $jitter_strength

                for warm_up in 10 30 50 70 90; do
                    echo $warm_up

                    job_dir="$results_dir/$inner_method/k_$k/iterations_$iterations/jitter_strength_$jitter_strength/warm_up_$warm_up"
                    mkdir -p $job_dir
                    cd $job_dir
                    qsub -q $queue -pe $pe \
                        "$base/xgboost_ablation_run.sh" \
                            --inner_method $inner_method \
                            --k $k \
                            --max_iter $iterations \
                            --jitter_strength $jitter_strength \
                            --warm_up $warm_up 
                    cd $base

                done
            done
        done 
    done
done