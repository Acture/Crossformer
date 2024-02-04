#!/bin/bash

run_command_for_dataset() {
	local dataset=$1
	local in_len=$2
	local predict_len=$3

	local python_command options logfile

	case "$dataset" in
	"ETTh1" | "ETTh2" | "ETTm1" | "ETTm2" | "weather")
		options="--seg_len 24 --learning_rate 1e-4 --itr 5"
		;;
	"electricity")
		options="--seg_len 24 --d_model 64 --d_ff 128 --n_heads 2 --learning_rate 5e-4 --lradj fixed --itr 5"
		;;
	"illness")
		options="--seg_len 6 --e_layers 2 --learning_rate 5e-4 --dropout 0.6 --itr 5"
		;;
	"traffic")
		options="--seg_len 6 --d_model 64 --d_ff 128 --n_heads 2 --learning_rate 1e-3 --itr 5"
		;;
	*)
		echo "Invalid dataset"
		return
		;;
	esac

	python_command="python main_crossformer.py --data $dataset --in_len $in_len --out_len $predict_len $options"
	logfile="logs/exp_${dataset}_${predict_len}_${in_len}.log"

	# Run the command and write output to log file
	echo "Running: $python_command"
	$python_command > $logfile
}

all_datasets="ETTh1 ETTh2 ETTm1 ETTm2 weather electricity illness traffic"

# Check if the first argument is "all"
if [[ $1 == "all" ]]; then
	# If so, use all_datasets
	datasets=$all_datasets
else
	# Otherwise, use all the arguments
	datasets="$@"
fi

for dataset in $datasets; do
	if [ "$dataset" == "illness" ]; then
		predict_lengths="24 60"
		in_lens="24 36 48 60 104 144"
	else
		predict_lengths="96 720"
		in_lens="24 48 96 192 336 720"
	fi

	for predict_len in $predict_lengths; do
		for in_len in $in_lens; do
			echo "Dataset: $dataset, Predict length: $predict_len, in_len: $in_len"
			run_command_for_dataset "$dataset" "$in_len" "$predict_len"
		done
	done
done
