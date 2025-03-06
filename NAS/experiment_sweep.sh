#!/usr/bin/env bash

timeout="30000"
max_tasks="500"
n_trials=5

queue="long@@cvrl"
pe="smp 1"

benchmarks=('SVMBenchmark' 'LRBenchmark' 'RandomForestBenchmark')
# benchmarks=('NNBenchmark')
# benchmarks=('XGBoostBenchmark')
datasets=('167119' '10101' '53' '146818' '146821' '9952' '146822' '31' '3917' '168912' '3' '12' '146212' '168911' '9981' '167120' '14965' '146606' '7592' '9977')
# datasets=('10101' '53' '146818' '146821' '9952' '146822' '31' '3917') # NNBenchmark
# methods=("bayes" "ncqs" "hom" "random" "smac")
methods=("bayes" "ncqs" "random" "smac" "tpe")
inner_methods=("bayes" "random" "smac" "tpe")
seeds=('' '42' '7' '101' '969535336' '666')

base="$(realpath $(pwd))"
results_dir="$base/ncqs_results"
script="$base/svm_ncqs_random.py"

# start a counter 
i=0

for benchmark in ${benchmarks[@]}; do
    echo $benchmark
    
    for dataset in ${datasets[@]}; do
        echo $dataset
        
        for method in ${methods[@]}; do
            job_dir="$results_dir/$benchmark/$dataset/$method/"
            
            echo $method
            echo $method | grep -q -E "(ncqs|hom)"
            
            if [ $? -eq 0 ]; then
                
                for inner_method in ${inner_methods[@]}; do
                    
                    for t in $(seq $n_trials); do
                        seed=${seeds[t]}
                        trial_dir="$job_dir/$inner_method/trial_$t"

                        python3 "$base/done_check.py" --results-file "$trial_dir/results.json" --num-trials $max_tasks
                        
                        if [ $? -eq 1 ]; then
                            # increment counter
                            # ((i=i+1)) 
                            mkdir -p $trial_dir
                            cd $trial_dir
                            # echo $trial_dir
                            qsub -q $queue -pe $pe \
                                "$base/run_random.sh" \
                                    --benchmark $benchmark \
                                    --dataset $dataset \
                                    --method $method \
                                    --inner-method $inner_method \
                                    --timeout $timeout \
                                    --max-tasks $max_tasks \
                                    --seed $seed
                            cd $base
                        fi
                    done
                done
            else
                for t in $(seq $n_trials); do
                    seed=${seeds[t]}
                    trial_dir="$job_dir/trial_$t"
                    
                    python3 "$base/done_check.py" --results-file "$trial_dir/results.json" --num-trials $max_tasks
                    
                    if [ $? -eq 1 ]; then
                        # increment counter
                        # ((i=i+1))
                        mkdir -p $trial_dir
                        cd $trial_dir
                        # echo $trial_dir
                        qsub -q $queue -pe $pe \
                            "$base/run_random.sh" \
                                --benchmark $benchmark \
                                --dataset $dataset \
                                --method $method \
                                --timeout $timeout \
                                --max-tasks $max_tasks \
                                --seed $seed
                        cd $base
                    fi
                done
            fi
        done
    done
done

echo $i
