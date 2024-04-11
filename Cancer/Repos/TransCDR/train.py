# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 15:38:32 2022

@author:Xiaoqiong Xia
@e-mail: 
@model: multi-omics model
@time:2021/9/15 16:33 
"""

import pandas as pd
from DataEncoding import DataEncoding
from model import TransCDR
import argparse

parser = argparse.ArgumentParser(description='CV10 of TransCDR')
parser.add_argument('--device', type=str, required=True, help='cpu, cuda, cuda:0')
parser.add_argument('--data_path', type=str, required=True, help='the data path of CV folder.')
parser.add_argument('--omics', type=str, required=True, help='expr, mutation, methylation, expr + mutation, expr + methylation, mutation + methylation, expr + mutation + methylation')
parser.add_argument('--input_dim_drug', type=int, required=True, help='seq_model: 768, graph_model: 300')
parser.add_argument('--lr', type=float, required=True, help='Learning rate')
parser.add_argument('--batch_size', type=int, required=True, help='Batch size test set')
parser.add_argument('--train_epoch', type=int, required=True, help='Number of epoch')
parser.add_argument('--drug_model', type=str, required=True, help='sequence, graph')
parser.add_argument('--modeldir', type=str, required=True, help='the dir of training results')
parser.add_argument('--seq_model', type=str, required=False, help='seyonec/ChemBERTa-zinc-base-v1, seyonec/PubChem10M_SMILES_BPE_450k, seyonec/ChemBERTa_zinc250k_v2_40k, seyonec/SMILES_tokenized_PubChem_shard00_160k, seyonec/PubChem10M_SMILES_BPE_180k, seyonec/PubChem10M_SMILES_BPE_396_250, seyonec/ChemBERTA_PubChem1M_shard00_155k, seyonec/BPE_SELFIES_PubChem_shard00_50k, seyonec/BPE_SELFIES_PubChem_shard00_160k')
parser.add_argument('--graph_model', type=str, required=False, help='gin_supervised_contextpred, gin_supervised_infomax, gin_supervised_edgepred, gin_supervised_masking')
parser.add_argument('--conformal_prediction', type=str, required=True, help='Whether to take pIC50 values or errors.')
args = parser.parse_args()

for i in range(10):
    train = pd.read_csv(args.data_path + f'/fold{i}/train.csv')
    test = pd.read_csv(args.data_path + f'/fold{i}/test.csv')
    val = pd.read_csv(args.data_path + f'/fold{i}/val.csv')
    
    config = {
        'device': args.device,
        'omics': args.omics, # expr + mutation + methylation
        'input_dim_drug': args.input_dim_drug,  # graph:300, seq:768, sequence + graph:1068
        'input_dim_rna': 18451,
        'input_dim_genetic': 735,
        'input_dim_mrna': 20617,
        'lr':args.lr, # learning rate
        'decay':0,
        'batch_size':args.batch_size, # batch size
        'train_epoch':args.train_epoch, # training epoch
        'drug_model': args.drug_model, # graph, sequence, sequence + graph FP
        'modeldir':args.modeldir + '/fold'+str(i),
        'seq_model':args.seq_model, # the pre-trained smiles model
        'graph_model':args.graph_model,  # the pre-trained graph model
        'conformal_prediction':args.conformal_prediction
    }
    data_encoding = DataEncoding(**config)
    train_set, test_set, val_set = data_encoding.encode(train, test, val)
    net = TransCDR(**config)
    net.train(train_drug=train_set,
              test_drug=test_set,
              val_drug=val_set)
    net.save_model()
