#!/usr/bin/env bash


dataset="bitcoinAlpha"
dataset="bitcoinOTC"
#dataset="epinions_truncated"
#dataset="slashdot_truncated"

mainfold="test_data_randomsplit"
logfold="log_64_32_32"
device="0"
device="1"

dim=64
lambda_list="4"
for idx in $(seq 0 9)
do
  for lambd in ${lambda_list}
  do
    for rept in $(seq 0 0)
    do
      echo "==========lambda${lambd}_epoch${idx}_rept${rept} $(date +"%Y%m%d %H:%M:%S")=========================="
      model_file="$logfold/lambda_randomsplit/${dataset}/${dataset}_${idx}_lambda${lambd}_rept${rept}.pkl"
      log_file="$logfold/lambda_randomsplit/${dataset}/${dataset}_${idx}_lambda${lambd}_rept${rept}.log"
      cuda_threads=$(ps -ef | grep "cuda_device ${device}" | wc -l)

      echo "${cuda_threads} running"
      while [ $cuda_threads -ge 5 ]
      do
          cuda_threads=$(ps -ef | grep "cuda_device ${device}" | wc -l)
	  sleep 5s
      done
      python main.py \
	    --cuda_device ${device} \
	    --lambda_structure ${lambd} \
	    --batch_size 1000 \
	    --test_interval 10 \
	    --network_file_name ${mainfold}/train_test/${dataset}/${dataset}_train${idx}.edgelist \
	    --test_network_file_name ${mainfold}/train_test/${dataset}/${dataset}_test${idx}.edgelist \
	    --feature_file_name ${mainfold}/features/${dataset}/${dataset}_train${idx}_features${dim}_tsvd.pkl \
	    --model_path ${model_file} \
	    --total_minibatches 1000 > $log_file 2>&1 &
      #device=$((1-$device))
      sleep 5s
    done
  done
done






:<<"EOF"
#dataset="slashdot_truncated"
dataset="epinions_truncated"

lambda_list="4"
device="-1"
for idx in $(seq 0 0)
do
  for lambd in ${lambda_list}
  do
    for rept in $(seq 0 0)
    do
      echo "==========lambda${lambd}_epoch${idx}_rept${rept} $(date +"%Y%m%d %H:%M:%S")=========================="
      model_file="log/lambda_randomsplit/${dataset}/${dataset}_${idx}_lambda${lambd}_rept${rept}.pkl"
      log_file="log/lambda_randomsplit/${dataset}/${dataset}_${idx}_lambda${lambd}_rept${rept}.log"
      cuda_threads=$(ps -ef | grep "cuda_device ${device}" | wc -l)

      echo "${cuda_threads} running"
      #while [ $cuda_threads -ge 9 ]
      #do
      #    cuda_threads=$(ps -ef | grep "cuda_device ${device}" | wc -l)
      #	  sleep 5s
      #done
      python model_lambda_adagrad_tsvd.py \
	    --cuda_device ${device} \
	    --loss2_regularization ${lambd} \
	    --batch_size 5000 \
	    --validation_interval 100 \
	    --patience 200 \
	    --network_file_name ${mainfold}/train_test/${dataset}/${dataset}_train${idx}.edgelist \
	    --val_network_file_name ${mainfold}/train_test/${dataset}/${dataset}_test${idx}.edgelist \
            --feature_file_name ${mainfold}/features/${dataset}/${dataset}_train${idx}_features64_tsvd.pkl \
	    --model_path ${model_file} \
	    --embedding_output_directory ${mainfold} \
	    --total_minibatches 4000 > $log_file 2>&1 &
      sleep 5s
    done
  done
done
EOF
