#!/bin/bash
python train.py \
--device 'cuda:2' \
--data_path '../../Data/result/combined/CV10' \
--omics 'expr + mutation + methylation' \
--input_dim_drug 1024 \
--lr 1e-4 \
--batch_size 128 \
--train_epoch 10 \
--drug_model 'FP' \
--modeldir './result/combined/CV10' \
--conformal_prediction 'False'

python train.py \
--device 'cuda:2' \
--data_path '../../Data/result/combined/CV10_cold_cell' \
--omics 'expr + mutation + methylation' \
--input_dim_drug 1024 \
--lr 1e-4 \
--batch_size 128 \
--train_epoch 10 \
--drug_model 'FP' \
--modeldir './result/combined/CV10_cold_cell' \
--conformal_prediction 'False'

python train.py \
--device 'cuda:2' \
--data_path '../../Data/result/combined/CV10_cold_drug' \
--omics 'expr + mutation + methylation' \
--input_dim_drug 1024 \
--lr 1e-4 \
--batch_size 128 \
--train_epoch 10 \
--drug_model 'FP' \
--modeldir './result/combined/CV10_cold_drug' \
--conformal_prediction 'False'

python train.py \
--device 'cuda:2' \
--data_path '../../Data/result/combined/CV10_cold_drug_cell' \
--omics 'expr + mutation + methylation' \
--input_dim_drug 1024 \
--lr 1e-4 \
--batch_size 128 \
--train_epoch 10 \
--drug_model 'FP' \
--modeldir './result/combined/CV10_cold_drug_cell' \
--conformal_prediction 'False'