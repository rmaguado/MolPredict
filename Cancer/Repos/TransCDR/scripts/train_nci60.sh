#!/bin/bash
python train.py \
--device 'cuda:1' \
--data_path '../../Data/result/nci60/CV10' \
--omics 'expr + mutation + methylation' \
--input_dim_drug 2092 \
--lr 1e-4 \
--BATCH_SIZE 128 \
--train_epoch 10 \
--pre_train 'True' \
--fusion_type 'encoder' \
--drug_encoder 'None' \
--drug_model 'sequence + graph + FP' \
--modeldir './result/nci60/CV10' \
--seq_model 'seyonec/PubChem10M_SMILES_BPE_450k' \
--graph_model 'gin_supervised_masking' \
--conformal_prediction 'False'

python train.py \
--device 'cuda:1' \
--data_path '../../Data/result/nci60/CV10_cold_cell' \
--omics 'expr + mutation + methylation' \
--input_dim_drug 2092 \
--lr 1e-4 \
--BATCH_SIZE 128 \
--train_epoch 10 \
--pre_train 'True' \
--fusion_type 'encoder' \
--drug_encoder 'None' \
--drug_model 'sequence + graph + FP' \
--modeldir './result/nci60/CV10_cold_cell' \
--seq_model 'seyonec/PubChem10M_SMILES_BPE_450k' \
--graph_model 'gin_supervised_masking' \
--conformal_prediction 'False'

python train.py \
--device 'cuda:1' \
--data_path '../../Data/result/nci60/CV10_cold_drug' \
--omics 'expr + mutation + methylation' \
--input_dim_drug 2092 \
--lr 1e-4 \
--BATCH_SIZE 128 \
--train_epoch 10 \
--pre_train 'True' \
--fusion_type 'encoder' \
--drug_encoder 'None' \
--drug_model 'sequence + graph + FP' \
--modeldir './result/nci60/CV10_cold_drug' \
--seq_model 'seyonec/PubChem10M_SMILES_BPE_450k' \
--graph_model 'gin_supervised_masking' \
--conformal_prediction 'False'

python train.py \
--device 'cuda:1' \
--data_path '../../Data/result/nci60/CV10_cold_drug_cell' \
--omics 'expr + mutation + methylation' \
--input_dim_drug 2092 \
--lr 1e-4 \
--BATCH_SIZE 128 \
--train_epoch 10 \
--pre_train 'True' \
--fusion_type 'encoder' \
--drug_encoder 'None' \
--drug_model 'sequence + graph + FP' \
--modeldir './result/nci60/CV10_cold_drug_cell' \
--seq_model 'seyonec/PubChem10M_SMILES_BPE_450k' \
--graph_model 'gin_supervised_masking' \
--conformal_prediction 'False'