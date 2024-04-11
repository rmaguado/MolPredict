# python3
# -*- coding:utf-8 -*-

"""
@author:野山羊骑士
@e-mail：thankyoulaojiang@163.com
@file：PycharmProject-PyCharm-DataEncoding.py
@time:2021/9/7 10:04 
"""
import numpy as np
import pandas as pd
import codecs
from subword_nmt.apply_bpe import BPE

class DataEncoding:
    def __init__(self,**config):
        self.config = config

    def _drug2emb_encoder(self,smile):
        '''get the token and mask of drugs'''
        
        vocab_path = './ESPF/drug_codes_chembl_freq_1500.txt'
        sub_csv = pd.read_csv('./ESPF/subword_units_map_chembl_freq_1500.csv')

        bpe_codes_drug = codecs.open(vocab_path)
        dbpe = BPE(bpe_codes_drug, merges=-1, separator='')

        idx2word_d = sub_csv['index'].values
        words2idx_d = dict(zip(idx2word_d, range(0, len(idx2word_d))))

        max_d = 50
        t1 = dbpe.process_line(smile).split()  # split
        try:
            i1 = np.asarray([words2idx_d[i] for i in t1])  # index
        except:
            i1 = np.array([0])

        l = len(i1)
        if l < max_d:
            i = np.pad(i1, (0, max_d - l), 'constant', constant_values=0)
            input_mask = ([1] * l) + ([0] * (max_d - l)) # the mask of true token is 1, the padding is 0
        else:
            i = i1[:max_d]
            input_mask = [1] * max_d

        return i, np.asarray(input_mask)

    def encode(self,traindata,testdata,valdata):
        cell_info = pd.read_csv("../../Data/processed/cell_info.csv")
        cell_info = cell_info.rename(columns={"cell_type":"cell_line"})
        # prepare data
        #drug_smiles = pd.concat([traindata, testdata, valdata], axis=0)
        #smile_encode = pd.Series(drug_smiles['smiles'].unique()).apply(self._drug2emb_encoder)
        #uniq_smile_dict = dict(zip(drug_smiles['smiles'].unique(),smile_encode))
        if (testdata is not None) & (valdata is not None):
            #traindata['drug_encoding'] = [uniq_smile_dict[i] for i in traindata['smiles']]
            #testdata['drug_encoding'] = [uniq_smile_dict[i] for i in testdata['smiles']]
            #valdata['drug_encoding'] = [uniq_smile_dict[i] for i in valdata['smiles']]
            if self.config['conformal_prediction'] == 'True':
                traindata = traindata.drop(columns={"Label"})
                testdata = testdata.drop(columns={"Label"})
                valdata = valdata.drop(columns={"Label"})
                data_label = 'Residual'
            else:
                data_label = 'pIC50'
            
            traindata = pd.merge(traindata, cell_info, how='left', on='cell_line')
            testdata = pd.merge(testdata, cell_info, how='left', on='cell_line')
            valdata = pd.merge(valdata, cell_info, how='left', on='cell_line')
            
            traindata = traindata.reset_index()
            testdata = testdata.reset_index()
            valdata = valdata.reset_index()
            
            traindata = traindata.rename(columns={data_label:'Label'})
            testdata = testdata.rename(columns={data_label:'Label'})
            valdata = valdata.rename(columns={data_label:'Label'})
            
            columns = ["index", "cell_line", "assay_name", "COSMIC_ID", "smiles", "Label"]
            traindata = traindata[columns]
            testdata = testdata[columns]
            valdata = valdata[columns]
            
            return traindata, testdata, valdata
        else:
            raise('Missing data.')
