#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH -o ./out/%j.txt
#SBATCH -e ./err/%j.txt

job_id="--j=$SLURM_JOB_ID"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --ex) experiment_id="--ex=$2"; shift ;;
        --plot) plot="--plot"; shift ;;
        --save-model) save_model="--save-model"; shift ;;
        --bsm) budget_scale_max="--bsm=$2"; shift ;;
        --lr) lr="--lr=$2"; shift ;;
        --ss) ss_tau="--ss=$2"; shift ;;
        --sm) sm_tau="--sm=$2"; shift ;;
        --reg) reg_lambda="--reg=$2"; shift ;;
        --bs) batch_size="--bs=$2"; shift ;;
        --e) epochs="--e=$2"; shift ;;
        --es) early_stopping="--es=$2"; shift ;;
        --ds) data_seed="--ds=$2"; shift ;;
        --ws) weights_seed="--ws=$2"; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

# echo "$job_id, $experiment_id
# echo "$plot, $save_model"
# echo "$budget_scale_max"
# echo "$lr, $ss_tau, $sm_tau, $reg_lambda"
# echo "$batch_size, $epochs, $early_stopping"
# echo "$data_seed, $weights_seed"

python3 run_adult_data.py \
 $job_id $experiment_id \
 $budget_scale_max \
 $lr $ss_tau $sm_tau $reg_lambda \
 $batch_size $epochs $early_stopping \
 $data_seed $weights_seed \
 $plot $save_model