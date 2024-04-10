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
from Step1_getData import GetData

class DataEncoding:
    def __init__(self,**config):
        self.config = config
    def _drug2emb_encoder(self,smile):
        vocab_path = "./ESPF/drug_codes_chembl_freq_1500.txt"
        sub_csv = pd.read_csv("./ESPF/subword_units_map_chembl_freq_1500.csv")

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
            input_mask = ([1] * l) + ([0] * (max_d - l))
        else:
            i = i1[:max_d]
            input_mask = [1] * max_d

        return i, np.asarray(input_mask)

    def encode(self,traindata,testdata,valdata):
        cell_info = pd.read_csv("../../Data/processed/cell_info.csv")
        cell_info = cell_info.rename(columns={"cell_type":"cell_line"})
        
        drug_smiles = pd.concat([traindata, testdata, valdata], axis=0)
        smile_encode = pd.Series(drug_smiles['smiles'].unique()).apply(self._drug2emb_encoder)
        uniq_smile_dict = dict(zip(drug_smiles['smiles'].unique(),smile_encode))

        traindata['drug_encoding'] = [uniq_smile_dict[i] for i in traindata['smiles']]
        testdata['drug_encoding'] = [uniq_smile_dict[i] for i in testdata['smiles']]
        valdata['drug_encoding'] = [uniq_smile_dict[i] for i in valdata['smiles']]
        if self.config['conformal_prediction'] == 'True':
                data_label = 'error'
            else:
                data_label = 'pIC50'
        
        traindata = pd.merge(traindata, cell_info, how='left', on='cell_line')
        testdata = pd.merge(testdata, cell_info, how='left', on='cell_line')
        valdata = pd.merge(valdata, cell_info, how='left', on='cell_line')
        
        traindata = traindata.reset_index()
        testdata = testdata.reset_index()
        valdata = valdata.reset_index()
        
        traindata['Label'] = traindata['pIC50']
        testdata['Label'] = testdata['pIC50']
        valdata['Label'] = valdata['pIC50']

        train_rnadata, test_rnadata = self.Getdata.getRna(
            traindata=traindata,
            testdata=testdata)
        train_rnadata = train_rnadata.T
        
        print("rnadata", len(train_rnadata))
        
        test_rnadata = test_rnadata.T
        train_rnadata.index = range(train_rnadata.shape[0])
        test_rnadata.index = range(test_rnadata.shape[0])

        return traindata, train_rnadata, testdata, test_rnadata

if __name__ == '__main__':
    obj = DataEncoding()
    traindata, testdata = obj.Getdata.ByCancer(random_seed= 1)

    traindata, train_rnadata, testdata, test_rnadata = obj.encode(
        traindata=traindata,
        testdata=testdata)

    print(traindata, train_rnadata, testdata, test_rnadata)
