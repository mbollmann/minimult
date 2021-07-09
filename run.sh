##########################################################################################
######## SETUP
# myenv multl
WORKDIR="/home/bollmann/data/subwords"
export CUDA_VISIBLE_DEVICES=99  # comment out to use CUDA (on lyria)
export seed=42

runexperiment() {
	# run pretraining
	python run_language_modeling.py \
	--train_data_file ${WORKDIR}/data/${6}.train.txt \
	--output_dir ${WORKDIR}/minimult-models/${1},${10} \
	--model_type bert \
	--mlm \
	--config_name configs/bert-small.json \
	--tokenizer_name ${WORKDIR}/tokenizer/${3}.json \
	--do_train \
	--do_eval \
	--per_gpu_train_batch_size ${4} \
	--num_train_epochs ${2} \
	--warmup_steps 50 \
	--logging_steps ${9} \
	--save_steps ${9} \
	--overwrite_output_dir \
	--block_size 128 \
	--line_by_line \
	--eval_data_file ${WORKDIR}/data/${6}.dev.txt \
	--per_gpu_eval_batch_size ${4} \
	--learning_rate ${7} \
	--weight_decay 0.01 \
	--adam_epsilon 1e-6 \
	--seed ${10} \
	--overwrite_cache \
	--gradient_accumulation_steps ${5} \
	${11}

	# evaluate multilinguality
	#python evaluate.py \
	#--model_name_or_path ${WORKDIR}/models/${10},${1} \
	#--eval_data_file ${WORKDIR}/corpora/eng_kingjames_10K.txt \
	#--exid ${1} \
	#--seed ${10} \
	#--modeltype bert \
	#--take_n_sentences ${8} \
	#--outfile ${WORKDIR}/results/kingjames10K.txt \
	#${11}
	# perplexity on dev
	python run_language_modeling.py \
	--train_data_file ${WORKDIR}/data/${6}.dev.txt \
	--model_type bert \
	--output_dir ${WORKDIR}/minimult-models/${1},${10} \
	--model_name_or_path ${WORKDIR}/minimult-models/${1},${10} \
	--mlm \
	--do_eval \
	--per_gpu_train_batch_size ${4} \
	--num_train_epochs ${2} \
	--warmup_steps 50 \
	--logging_steps ${9} \
	--save_steps ${9} \
	--overwrite_output_dir \
	--block_size 128 \
	--line_by_line \
	--eval_data_file ${WORKDIR}/data/${6}.dev.txt \
	--per_gpu_eval_batch_size ${4} \
	--learning_rate ${7} \
	--weight_decay 0.01 \
	--adam_epsilon 1e-6 \
	--seed ${10} \
	--overwrite_cache \
	--gradient_accumulation_steps ${5} \
	--eval_output_file ${WORKDIR}/minimult-results/perpl_mask_only.${6}.dev.txt \
	--replacement_probs 1.0,0.0,0.0 \
	${11}
}

##########################################################################################
######## EXPERIMENTS

#for seed in 0 42 43 100 101
#do

seed=42

# runexperiment 0 100 "orig.multi_Cyrillic" 256 1 "bg" 2e-3 -1 135 ${seed} ""
#               1 2   3                     4   5 6    7    8  9   10      11

#  1: experiment ID
#  2: no. epochs
#  3: tokenizer name
#  4: batch size
#  5: gradient accumulation steps ?
#  6: input file
#  7: learning rate
#  8: [[only for evaluate.py -- no. sentences]]
#  9: logging/saving steps
# 10: random seed
# 11: additional CLI parameters

runexperiment 0 100 "orig.multi_Cyrillic" 256 1 "bg" 2e-3 -1 135 ${seed} ""


#	runexperiment 1 100 small 256 1 "" 2e-3 -1 100000 ${seed} "--language_specific_positions"
#	runexperiment 2 100 small 256 1 "" 2e-3 -1 100000 ${seed} "--shift_special_tokens"
#	runexperiment 3 100 small 256 1 "" 2e-3 -1 100000 ${seed} "--invert_order"
#	runexperiment 4 100 small 256 1 "" 2e-3 -1 100000 ${seed} "--do_not_replace_with_random_words"
#	runexperiment 5 100 small 256 1 "" 2e-3 -1 100000 ${seed} "--language_specific_positions --shift_special_tokens"
#	runexperiment 6 100 small 256 1 "" 2e-3 -1 100000 ${seed} "--language_specific_positions --do_not_replace_with_random_words"

#	runexperiment 7 100 small 256 1 "" 2e-3 -1 100000 ${seed} "--shift_special_tokens --do_not_replace_with_random_words"
#	runexperiment 8 100 small 256 1 "" 2e-3 -1 100000 ${seed} "--language_specific_positions --shift_special_tokens --do_not_replace_with_random_words"
#	runexperiment 9 100 small 256 1 "" 2e-3 -1 100000 ${seed} "--language_specific_positions --shift_special_tokens --do_not_replace_with_random_words --invert_order"

#	runexperiment 18 1 small 2 1 _2lines 2e-3 -1 100000 ${seed} ""
#	runexperiment 19 1 small 2 1 _2lines 2e-3 -1 100000 ${seed} "--language_specific_positions"
#	runexperiment 21 200 small 256 1 "" 2e-3 -1 100000 ${seed} "--no_parallel_data"
#	runexperiment 21b 200 small 256 1 "" 2e-3 -1 100000 ${seed} "--no_parallel_data --language_specific_positions"

#	runexperiment 15 100 base 16 16 "" 1e-4 -1 100000 ${seed} ""
#	runexperiment 16 100 base 16 16 "" 1e-4 -1 100000 ${seed} "--language_specific_positions"
#	runexperiment 17 100 base 16 16 "" 1e-4 -1 675 ${seed} "--language_specific_positions --shift_special_tokens --do_not_replace_with_random_words"

#	runexperiment 30 100 small 256 1 "" 2e-3 -1 100000 ${seed} "--replace_with_nn 1 --replacement_probs 0.5,0.2,0.6 --vecmap ${WORKDIR}/vecmap/vectorsmapped.vec"
#	runexperiment 31 100 small 256 1 "" 2e-3 -1 100000 ${seed} "--replace_with_nn 5 --replacement_probs 0.5,0.2,0.6 --vecmap ${WORKDIR}/vecmap/vectorsmapped.vec"
#done
