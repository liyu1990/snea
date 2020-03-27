#!/usr/bin/env bash


mainfold="test_data_randomsplit"
logfold="log_64_32_32"
dim=64

:<<"EOF"
#dataset="bitcoinAlpha"
dataset="bitcoinOTC"

#device="0"
device="1"
lambda_list="4"
weights4no="0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5"

for idx in $(seq 0 9)
do
  for lambd in ${lambda_list}
  do
    for w4no in ${weights4no} 
    do
    for rept in $(seq 0 1)
    do
      echo "==========lambda${lambd}_epoch${idx}_rept${rept} $(date +"%Y%m%d %H:%M:%S")=========================="
      model_file="$logfold/lambda_randomsplit/${dataset}/${dataset}_${idx}_lambda${lambd}_rept${rept}_w4no${w4no}.pkl"
      log_file="$logfold/lambda_randomsplit/${dataset}/${dataset}_${idx}_lambda${lambd}_rept${rept}_w4no${w4no}.log"
      cuda_threads=$(ps -ef | grep "cuda_device ${device}" | wc -l)

      echo "${cuda_threads} running"
      while [ $cuda_threads -ge 8 ]
      do
          cuda_threads=$(ps -ef | grep "cuda_device ${device}" | wc -l)
	  sleep 5s
      done
      python main.py \
	    --cuda_device ${device} \
	    --lambda_structure ${lambd} \
	    --class_weight_no ${w4no} \
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
done
EOF





#:<<"EOF"
dataset="slashdot_truncated"
#dataset="epinions_truncated"

lambda_list="4"
device="-1"
w4no="0.35"
for idx in $(seq 0 0)
do
  for lambd in ${lambda_list}
  do
    for rept in $(seq 0 0)
    do
      echo "==========lambda${lambd}_epoch${idx}_rept${rept} $(date +"%Y%m%d %H:%M:%S")=========================="
      model_file="$logfold/lambda_randomsplit/${dataset}/${dataset}_${idx}_lambda${lambd}_rept${rept}_w4no${w4no}.pkl"
      log_file="$logfold/lambda_randomsplit/${dataset}/${dataset}_${idx}_lambda${lambd}_rept${rept}_w4no${w4no}.log"
      cuda_threads=$(ps -ef | grep "cuda_device ${device}" | wc -l)

      echo "${cuda_threads} running"
      python main.py \
	    --cuda_device ${device} \
	    --lambda_structure ${lambd} \
	    --class_weight_no ${w4no} \
	    --batch_size 5000 \
	    --test_interval 100 \
	    --network_file_name ${mainfold}/train_test/${dataset}/${dataset}_train${idx}.edgelist \
	    --test_network_file_name ${mainfold}/train_test/${dataset}/${dataset}_test${idx}.edgelist \
            --feature_file_name ${mainfold}/features/${dataset}/${dataset}_train${idx}_features64_tsvd.pkl \
	    --model_path ${model_file} \
	    --total_minibatches 4000 > $log_file 2>&1 &
      sleep 5s
    done
  done
done
#EOF
