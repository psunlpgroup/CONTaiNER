export G=$1
export GPU=$2
export WAY=$3
export SHOT=$4
export way=${WAY}
export shot=${SHOT}
echo $shot $way
export SAVED_MODEL_DIR=fewnerd_model
export finetune_loss=KL
export is_viterbi=viterbi

## training with toy evaluation for sanity check
#export i=0
#python src/container.py --data_dir data/few-nerd/${G} --labels-train data/few-nerd/${G}/labels_train.txt --labels-test data/few-nerd/${G}/labels_test.txt --config_name bert-base-uncased --model_name_or_path bert-base-uncased --saved_model_dir saved_models/few-nerd/${G}/${SAVED_MODEL_DIR} --output_dir outputs/few-nerd/${G}/${finetune_loss}_${is_viterbi}_final/${G}-${way}-${shot}/${i} --support_path support_test_${way}_${shot}/${i} --test_path query_test_${way}_${shot}/${i} --n_shots ${shot} --max_seq_length 128 --embedding_dimension 128 --num_train_epochs 1 --train_batch_size 32 --seed 1 --do_train --do_predict --select_gpu ${GPU} --training_loss KL --finetune_loss ${finetune_loss} --evaluation_criteria euclidean

## evaluation
for ((i=0;i<50;i++))
  do
  echo $shot $way $i
  python src/container.py --data_dir data/few-nerd/${G} --labels-train data/few-nerd/${G}/labels_train.txt --labels-test data/few-nerd/${G}/labels_test.txt --config_name bert-base-uncased --model_name_or_path bert-base-uncased --saved_model_dir saved_models/few-nerd/${G}/${SAVED_MODEL_DIR} --output_dir outputs/few-nerd/${G}/${finetune_loss}_${is_viterbi}_final/${G}-${way}-${shot}/${i} --support_path support_test_${way}_${shot}/${i} --test_path query_test_${way}_${shot}/${i} --n_shots ${shot} --max_seq_length 128 --embedding_dimension 128 --num_train_epochs 1 --train_batch_size 32 --seed 1 --do_predict --select_gpu ${GPU} --training_loss KL --finetune_loss ${finetune_loss} --evaluation_criteria euclidean_hidden_state --learning_rate 5e-5 --learning_rate_finetuning 5e-5 --consider_mutual_O --temp_trans 0.01 --silent
  done