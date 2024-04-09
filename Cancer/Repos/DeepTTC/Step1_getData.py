# python3
# -*- coding:utf-8 -*-

"""
@author:野山羊骑士
@e-mail: thankyoulaojiang@163.com
@file: PycharmProject-PyCharm-Step1_getData.py
@time:2021/8/12 15:48 
"""
import os
import sys
import csv
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split

class GetData():
    def __init__(self):
        self.pairfile = './GDSC_data/GDSC2_fitted_dose_response_25Feb20.xlsx'
        self.drugfile = "./GDSC_data/Drug_listTue_Aug10_2021.csv"
        self.rnafile =  './GDSC_data/Cell_line_RMA_proc_basalExp.txt'
        self.smilefile = './GDSC_data/smile_inchi.csv'
        self.drug_thred = './GDSC_data/IC50_thred.txt'

    def getDrug(self):
        drugdata = pd.read_csv(self.smilefile,index_col=0)
        return drugdata

    def _filter_pair(self,drug_cell_df):
        print("#"*100)
        print("step1 filtering cell lines....")
        rnadata = pd.read_csv(self.rnafile,sep='\t')
        rna_ids = [x.split("DATA.")[1] for x in list(rnadata.columns) if "DATA." in x]
        print(drug_cell_df.shape)
        drug_cell_df = drug_cell_df[drug_cell_df['COSMIC_ID'].astype(str).isin(rna_ids)]
        
        print(drug_cell_df.shape)

        print("step2 filtering drugs....")
        pub_df = pd.read_csv(self.drugfile)
        pub_df = pub_df.dropna(subset=['PubCHEM'])
        pub_df = pub_df[(pub_df['PubCHEM'] != 'none') & (pub_df['PubCHEM'] != 'several')]
        print(drug_cell_df.shape)
        drug_cell_df = drug_cell_df[drug_cell_df['DRUG_ID'].isin(pub_df['drug_id'])]
        print(drug_cell_df.shape)
        return drug_cell_df

    def _stat_cancer(self,drug_cell_df):
        print("#" * 50)
        cancer_num = drug_cell_df['TCGA_DESC'].value_counts().shape[0]
        print('#\t Total Cancer types: {}'.format(cancer_num))
        min_cancer_drug = min(drug_cell_df['TCGA_DESC'].value_counts())
        max_cancer_drug = max(drug_cell_df['TCGA_DESC'].value_counts())
        mean_cancer_drug = np.mean(drug_cell_df['TCGA_DESC'].value_counts())
        print('#\t The smallest cancer type corresponds to {} drugs'.format(min_cancer_drug))
        print('\t The most corresponds to {} drugs'.format(max_cancer_drug))
        print('\t Corresponding to {} drugs on average'.format(mean_cancer_drug))

    def _stat_cell(self, drug_cell_df):
        print("#" * 50)
        cell_num = drug_cell_df['COSMIC_ID'].value_counts().shape[0]
        print('#\t The cell lines used are: {}'.format(cell_num))
        min_drug = min(drug_cell_df['COSMIC_ID'].value_counts())
        max_drug = max(drug_cell_df['COSMIC_ID'].value_counts())
        mean_drug = np.mean(drug_cell_df['COSMIC_ID'].value_counts())
        print('#\t The least number of drugs corresponds to {} cell lines'.format(min_drug))
        print('\t The most corresponds to {} cell lines'.format(max_drug))
        print('\t The average number corresponds to {} cell lines'.format(mean_drug))

    def _stat_drug(self, drug_cell_df):
        print("#" * 50)
        drug_num = drug_cell_df['DRUG_ID'].value_counts().shape[0]
        print('#\t The drugs used are: {}'.format(drug_num))
        min_cell = min(drug_cell_df['DRUG_ID'].value_counts())
        max_cell = max(drug_cell_df['DRUG_ID'].value_counts())
        mean_cell = np.mean(drug_cell_df['DRUG_ID'].value_counts())
        print('#\t The least number of drugs corresponds to {} cell lines'.format(min_cell))
        print('\t The most corresponds to {} cell lines'.format(max_cell))
        print('\t The average number corresponds to {} cell lines'.format(mean_cell))

    def _split(self,df,col,ratio,random_seed):

        col_list = df[col].value_counts().index
        train_data = pd.DataFrame()
        test_data = pd.DataFrame()

        for instatnce in col_list:
            sub_df = df[df[col] == instatnce]
            sub_df = sub_df[['DRUG_ID', 'COSMIC_ID','TCGA_DESC', 'LN_IC50']]
            ## 按照 col 来拆分数据集 ##
            ## 对于任意一个 instance，1 - ratio 的用于训练，10=test，10=validation
            sub_train, sub_test = train_test_split(sub_df, test_size=ratio,random_state=random_seed)
            if train_data.shape[0] == 0:
                train_data = sub_train
                test_data = sub_test
            else:
                train_data = pd.concat([train_data, sub_train])
                test_data = pd.concat([test_data, sub_test])
        print('#' * 50)
        print('#\t Total data pairs:{}'.format(df.shape[0]))
        print('#\t Cut the data according to {}. For each instance, the data of {} is used for training and the data of {} is used for verification.'.format(col,(1-ratio),ratio))
        print('#\t The training data has：{}'.format(train_data.shape[0]))
        print('#\t The test data has：{}'.format(test_data.shape[0]))

        return train_data,test_data

    def ByCancer(self,random_seed):

        # 理解作者的意思就是按照 癌症类型，随机选95的作为训练
        # 评价没有癌症的准确性，评价不同药物的准确性

        drug_cell_df = pd.read_excel(self.pairfile)
        self._stat_drug(drug_cell_df)
        self._stat_cell(drug_cell_df)
        self._stat_cancer(drug_cell_df)
        drug_cell_df = self._filter_pair(drug_cell_df)

        drug_cell_df = drug_cell_df.head(10000)
        self._stat_drug(drug_cell_df)
        self._stat_cell(drug_cell_df)
        self._stat_cancer(drug_cell_df)

        print(drug_cell_df['TCGA_DESC'].value_counts())

        train_data, test_data = self._split(df=drug_cell_df, col='TCGA_DESC',
                                            ratio=0.2,random_seed=random_seed)

        return train_data, test_data

    def ByDrug(self):
        drug_cell_df = pd.read_excel(self.pairfile)
        self._stat_drug(drug_cell_df)
        self._stat_cell(drug_cell_df)
        self._stat_cancer(drug_cell_df)
        drug_cell_df = self._filter_pair(drug_cell_df)
        self._stat_drug(drug_cell_df)
        self._stat_cell(drug_cell_df)
        self._stat_cancer(drug_cell_df)

        train_data,test_data = self._split(df=drug_cell_df,col='DRUG_ID',ratio=0.2)

        return train_data,test_data

    def ByCell(self):
        drug_cell_df = pd.read_excel(self.pairfile)
        self._stat_drug(drug_cell_df)
        self._stat_cell(drug_cell_df)
        self._stat_cancer(drug_cell_df)
        drug_cell_df = self._filter_pair(drug_cell_df)
        self._stat_drug(drug_cell_df)
        self._stat_cell(drug_cell_df)
        self._stat_cancer(drug_cell_df)

        train_data, test_data = self._split(df=drug_cell_df, col='COSMIC_ID', ratio=0.2)

        return train_data, test_data

    def MissingData(self):
        drug_cell_df = pd.read_excel(self.pairfile)
        self._stat_drug(drug_cell_df)
        self._stat_cell(drug_cell_df)
        self._stat_cancer(drug_cell_df)
        drug_cell_df = self._filter_pair(drug_cell_df)
        self._stat_drug(drug_cell_df)
        self._stat_cell(drug_cell_df)
        self._stat_cancer(drug_cell_df)

        cell_list = drug_cell_df['COSMIC_ID'].value_counts().index
        drug_list = drug_cell_df['DRUG_ID'].value_counts().index

        all_df = pd.DataFrame()
        dup_drug = []
        [dup_drug.extend([i]*len(cell_list)) for i in drug_list]
        all_df['DRUG_ID'] = dup_drug

        dup_cell = []
        for i in range(len(drug_list)):
            dup_cell.extend(cell_list)
        all_df['COSMIC_ID'] = dup_cell

        all_df['ID'] = all_df['DRUG_ID'].astype(str).str.cat(all_df['COSMIC_ID'].astype(str),sep='_')
        drug_cell_df['ID'] = drug_cell_df['DRUG_ID'].astype(str).str.cat(drug_cell_df['COSMIC_ID'].astype(str),sep='_')
        MissingData = all_df[~all_df['ID'].isin(drug_cell_df['ID'])]

        print("#"*100)
        print('Use {} drugs and {} cell lines'.format(len(drug_list),len(cell_list)))
        print('Theoretically, if each drug affects all cell lines, there should be {} Pairs'.format(len(drug_list)*len(cell_list)))
        print('However, some drugs and cell lines have not been tested. There are {} Pairs'.format(MissingData.shape[0]))

        # drug_cell_df = drug_cell_df[['COSMIC_ID', 'TCGA_DESC']].drop_duplicates()
        # cell2cancer_dict = pd.Series(list(drug_cell_df['TCGA_DESC']), index=drug_cell_df['COSMIC_ID'])

        return drug_cell_df,MissingData

    def _LeaveOut(self,df,col,ratio=0.8,random_num=1):
        random.seed(random_num)
        col_list = list(set(df[col]))
        col_list = list(col_list)

        sub_start = int(len(col_list)/5)*random_num
        if random_num==4:
            sub_end = len(col_list)
        else:
            sub_end = int(len(col_list)/5)*(random_num+1)

        # leave_instatnce = random.sample(col_list,int(len(col_list)*ratio))
        leave_instatnce = list(set(col_list)- set(col_list[sub_start:sub_end]))

        df = df[['DRUG_ID', 'COSMIC_ID', 'TCGA_DESC', 'LN_IC50']]
        train_data = df[df[col].isin(leave_instatnce)]
        test_data = df[~df[col].isin(leave_instatnce)]

        print('#' * 50)
        print(len(col_list))
        print(len(set(list(train_data[col]))))
        print(len(set(list(test_data[col]))))
        print('#\t Total of data pairs: {}, leave out method'.format(df.shape[0]))
        print('#\t Divide the data according to {}, and for each instance, train the data of {}'.format(col, ratio))
        print('#\t Training data：{}'.format(train_data.shape[0]))
        print('#\t Testing data：{}'.format(test_data.shape[0]))

        return train_data,test_data

    def Cell_LeaveOut(self,random):
        drug_cell_df = pd.read_excel(self.pairfile)
        self._stat_drug(drug_cell_df)
        self._stat_cell(drug_cell_df)
        self._stat_cancer(drug_cell_df)
        drug_cell_df = self._filter_pair(drug_cell_df)
        self._stat_drug(drug_cell_df)
        self._stat_cell(drug_cell_df)
        self._stat_cancer(drug_cell_df)

        traindata,testdata = self._LeaveOut(df=drug_cell_df,col='COSMIC_ID',ratio=0.8,random_num=random)

        return traindata,testdata

    def Drug_LeaveOut(self,random):
        drug_cell_df = pd.read_excel(self.pairfile)
        self._stat_drug(drug_cell_df)
        self._stat_cell(drug_cell_df)
        self._stat_cancer(drug_cell_df)
        drug_cell_df = self._filter_pair(drug_cell_df)
        self._stat_drug(drug_cell_df)
        self._stat_cell(drug_cell_df)
        self._stat_cancer(drug_cell_df)

        traindata, testdata = self._LeaveOut(df=drug_cell_df, col='DRUG_ID', ratio=0.8,random_num=random)

        return traindata, testdata

    def Drug_Thred(self):
        thred_data = pd.read_csv(self.drug_thred,sep='\t')
        thred_df = thred_data.T
        thred_df['drug_name'] =thred_df.index
        thred_df['threds'] = thred_df[0]
        thred_df = thred_df.drop(0,axis=1)
        thred_df.loc['VX-680','drug_name'] = 'Tozasertib'
        thred_df.loc['Mitomycin C','drug_name'] = 'Mitomycin-C'
        thred_df.loc['HG-6-64-1', 'drug_name'] = 'HG6-64-1'
        thred_df.loc['BAY 61-3606', 'drug_name'] = 'BAY-61-3606'
        thred_df.loc['Zibotentan, ZD4054', 'drug_name'] = 'Zibotentan'
        thred_df.loc['PXD101, Belinostat', 'drug_name'] = 'Belinostat'
        thred_df.loc['NU-7441', 'drug_name'] = 'NU7441'
        thred_df.loc['BIRB 0796', 'drug_name'] = 'BIRB-796'
        thred_df.loc['Nutlin-3a', 'drug_name'] = 'Nutlin-3a (-)'
        thred_df.loc['AZD6482.1', 'drug_name'] = 'AZD6482'
        thred_df.loc['BMS-708163.1', 'drug_name'] = 'BMS-708163'
        thred_df.loc['BMS-536924.1', 'drug_name'] = 'BMS-536924'
        thred_df.loc['GSK269962A.1', 'drug_name'] = 'GSK269962A'
        thred_df.loc['SB-505124', 'drug_name'] = 'SB505124'
        thred_df.loc['JQ1.1', 'drug_name'] = 'JQ1'
        thred_df.loc['UNC0638.1', 'drug_name'] = 'UNC0638'
        thred_df.loc['CHIR-99021.1', 'drug_name'] = 'CHIR-99021'
        thred_df.loc['piperlongumine', 'drug_name'] = 'Piperlongumine'
        thred_df.loc['PLX4720 (rescreen)', 'drug_name'] = 'PLX4720'
        thred_df.loc['Afatinib (rescreen)', 'drug_name'] = 'Afatinib'
        thred_df.loc['Olaparib.1', 'drug_name'] = 'Olaparib'
        thred_df.loc['AZD6244.1', 'drug_name'] = 'AZD6244'
        thred_df.loc['Bicalutamide.1', 'drug_name'] = 'Bicalutamide'
        thred_df.loc['RDEA119 (rescreen)', 'drug_name'] = 'RDEA119'
        thred_df.loc['GDC0941 (rescreen)', 'drug_name'] = 'GDC0941'
        thred_df.loc['MLN4924 ', 'drug_name'] = 'MLN4924'
        # only one I-BET 151

        drug_info = pd.read_csv(self.drugfile)
        drugname2drugid = {}
        drugid2pubchemid = {}
        for idx,row in drug_info.iterrows():
            name = row['Name']
            drug_id = row['drug_id']
            pub_id = row['PubCHEM']
            drugname2drugid[name] = drug_id
            drugid2pubchemid[drug_id] = pub_id

        drug_info_filter_name = drug_info.dropna(subset=['Synonyms'])
        for idx,row in drug_info_filter_name.iterrows():
            name = row['Name']
            pub_id = row['PubCHEM']
            drug_id = row['drug_id']
            drugname2drugid[name] = drug_id
            Synonyms_list = row['Synonyms'].split(', ')
            for drug in Synonyms_list:
                drugname2drugid[drug] = drug_id

        drugid2thred = {}
        for idx,row in thred_df.iterrows():
            name = row['drug_name']
            thred = row['threds']
            if name in drugname2drugid:
                drugid2thred[drugname2drugid[name]] = thred

        id_li = []
        PubChem_li =[]
        thred_li =[]
        for i in drugid2thred:
            id_li.append(i)
            PubChem_li.append(drugid2pubchemid[i])
            thred_li.append(drugid2thred[i])

        # data = pd.DataFrame()
        # data['Drug_id'] = id_li
        # data['PubChem'] = PubChem_li
        # data['Thred'] = thred_li
        #
        # print(data)
        # data.to_csv('Drug_Thred.csv')
        drug_list = [drugname2drugid[i] for i in list(thred_df['drug_name']) if i in drugname2drugid]

        return drug_list,drugid2thred

    def _split_no_balance_binary(self,df,col,ratio,random_seed):

        col_list = df[col].value_counts().index
        train_data = pd.DataFrame()
        test_data = pd.DataFrame()

        for instatnce in col_list:
            sub_df = df[df[col] == instatnce]
            sub_df = sub_df[['DRUG_ID', 'COSMIC_ID','TCGA_DESC', 'LN_IC50','Binary_IC50']]
            ## 按照 col 来拆分数据集 ##
            ## 对于任意一个 instance，1 - ratio 的用于训练，10=test，10=validation
            sub_train, sub_test = train_test_split(sub_df, test_size=ratio,
                                                   random_state=random_seed)
            if train_data.shape[0] == 0:
                train_data = sub_train
                test_data = sub_test
            else:
                train_data = pd.concat([train_data, sub_train])
                test_data = pd.concat([test_data, sub_test])
        print('#' * 50)
        print('#\t Total data pairs: {}'.format(df.shape[0]))
        print('#\t Cut the data according to {}. For each instance, the data of {} is used for training and the data of {} is used for verification.'.format(col,(1-ratio),ratio))
        print('#\t training data：{}'.format(train_data.shape[0]))
        print('#\t testing data：{}'.format(test_data.shape[0]))

        return train_data,test_data

    def _split_balance_binary(self,df,col,ratio,random_seed):

        col_list = df[col].value_counts().index

        pos_data = df[df[col]==1]
        neg_data = df[df[col]==0]

        down_pos_data = pos_data.loc[random.sample(list(pos_data.index),neg_data.shape[0])]

        combine_data = pd.concat([neg_data,down_pos_data])


        combine_data = combine_data[['DRUG_ID', 'COSMIC_ID','TCGA_DESC', 'LN_IC50','Binary_IC50']]

        train_data, test_data = train_test_split(combine_data, test_size=ratio,
                                                   random_state=random_seed)

        print('#' * 50)
        print('#\t total data pairs：{}'.format(df.shape[0]))
        print('#\t Construct a balanced data set, {} is a sample greater than -2, {} is a sample less than -2, select {} samples of 1:1 each'.format(
            pos_data.shape[0],neg_data.shape[0],neg_data.shape[0]))
        print('#\t Cut the data according to {}. For each instance, the data of {} is used for training and the data of {} is used for verification.'.format(
            col,(1-ratio),ratio))
        print('#\t train data：{}'.format(train_data.shape[0]))
        print('#\t test data：{}'.format(test_data.shape[0]))

        return train_data,test_data


    def ByBinary(self,random_num):
        drug_cell_df = pd.read_excel(self.pairfile)
        self._stat_drug(drug_cell_df)
        self._stat_cell(drug_cell_df)
        self._stat_cancer(drug_cell_df)
        drug_cell_df = self._filter_pair(drug_cell_df)
        self._stat_drug(drug_cell_df)
        self._stat_cell(drug_cell_df)
        self._stat_cancer(drug_cell_df)

        drug_list, drugid2thred = self.Drug_Thred()
        ##################################################
        # 按照每种药物得阈值,第一种，直接过滤
        Binary_Drug_list = []
        drug_cell_df = drug_cell_df[drug_cell_df['DRUG_ID'].isin(drug_list)]

        # print(drug_cell_df['DRUG_ID'].value_counts().shape)
        for idx,row in drug_cell_df.iterrows():
            drug_name = row['DRUG_NAME']
            drug_id = row['DRUG_ID']
            ic50 = row['LN_IC50']
            if (ic50 > drugid2thred[drug_id]):
                Binary_Drug_list.append(1)
            else:
                Binary_Drug_list.append(0)
        # 数量：2811*2 = Train * 4497 + Test 1125
        drug_cell_df['Binary_IC50'] = Binary_Drug_list
        ############################################################################
        # 第二种，补充-2的阈值
        # Binary_Drug_list = []
        #
        # print(drug_cell_df['DRUG_ID'].value_counts().shape)
        # for idx, row in drug_cell_df.iterrows():
        #     drug_name = row['DRUG_NAME']
        #     drug_id = row['DRUG_ID']
        #     ic50 = row['LN_IC50']
        #     if drug_id in drug_list:
        #         if ic50 > drugid2thred[drug_id]:
        #             Binary_Drug_list.append(1)
        #         else:
        #             Binary_Drug_list.append(0)
        #     else:
        #         if ic50 > -2:
        #             Binary_Drug_list.append(1)
        #         else:
        #             Binary_Drug_list.append(0)
        # drug_cell_df['Binary_IC50'] = Binary_Drug_list

        ############################################################################
        # 第三种 直接使用-2的阈值
        # Binary_IC50_list = []
        # for ic50 in drug_cell_df['LN_IC50']:
        #     if ic50 > -2:
        #         Binary_IC50_list.append(1)
        #     else:
        #         Binary_IC50_list.append(0)
        # drug_cell_df['Binary_IC50'] = Binary_IC50_list
        # 数量：9102*2 = Train 14571 + Test 3643

        #############################################################################
        # print(drug_cell_df['Binary_IC50'].value_counts())
        train_data, test_data = self._split_balance_binary(df=drug_cell_df, col='Binary_IC50',
                                                   ratio=0.2,random_seed=random_num)
        print(train_data,test_data)

        return train_data,test_data

    def getRna(self,traindata,testdata):
        rnadata =  pd.read_csv(self.rnafile,sep='\t')
        
        train_rnaid = list(traindata['COSMIC_ID'])
        test_rnaid = list(testdata['COSMIC_ID'])
        print("not found rna:",len(['DATA.'+str(i) for i in train_rnaid if 'DATA.' + str(i) not in rnadata.columns]))
        train_rnaid = ['DATA.'+str(i) for i in train_rnaid if 'DATA.' + str(i) in rnadata.columns]
        test_rnaid = ['DATA.' +str(i) for i in test_rnaid if 'DATA.' + str(i) in rnadata.columns]

        print("train_rna:", len(train_rnaid))
        print("test_rna:", len(train_rnaid))
        
        train_rnadata = rnadata[train_rnaid]
        test_rnadata = rnadata[test_rnaid]

        return train_rnadata,test_rnadata



if __name__ == '__main__':
    obj = GetData()
