for arg in "$@"
do
	for predict_len in 96 720
	do
		for in_len in 24 48 96 192 336 720
		do
		   echo "Dataset: $arg, Input size: $in_len, Predict size: $predict_len"
		   python main_crossformer.py\
		   --data $arg\
		   --in_len $in_len\
		   --out_len $predict_len\
		   --seg_len 24\
		   --d_model 64\
		   --d_ff 128\
		   --n_heads 2\
		   --learning_rate 5e-5\
		   --lradj fixed\
		   --itr 5\ > "logs/exp_"$arg"_"$predict_len"_"$in_len.log
		done
	done
	python log_extractor.py $arg".*" -s $arg".csv"
done
