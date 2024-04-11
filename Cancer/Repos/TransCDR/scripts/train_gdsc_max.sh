#!/bin/bash
python train.py \
--device 'cuda:0' \
--data_path '../../Data/result/gdsc/CV10' \
--omics 'expr + mutation + methylation' \
--input_dim_drug 2092 \
--lr 1e-4 \
--batch_size 128 \
--train_epoch 10 \
--drug_model 'sequence + graph + FP' \
--modeldir './result/gdsc_splits/CV10' \
--seq_model 'seyonec/PubChem10M_SMILES_BPE_450k' \
--graph_model 'gin_supervised_masking' \
--conformal_prediction 'False'
