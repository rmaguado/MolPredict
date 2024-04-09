# python3
# -*- coding:utf-8 -*-

"""
@author:野山羊骑士
@e-mail：thankyoulaojiang@163.com
@file：PycharmProject-PyCharm-step1_drugGet.py
@time:2021/8/10 15:48 
@从 pubchem数据库中，根据id查找smile
"""

import sys
import pandas as pd
import pubchempy as pcp
from tqdm import tqdm

pub_file = sys.argv[1]
pub_df = pd.read_csv(pub_file)
pub_df = pub_df.dropna(subset=['PubCHEM'])
pub_df = pub_df[(pub_df['PubCHEM']!='none') & (pub_df['PubCHEM']!='several')]

smile_list = []
inchi_list = []
for idx,row in tqdm(pub_df.iterrows(), total=len(pub_df)):
    pubid = row['PubCHEM'].split(',')[0]
    compound = pcp.Compound.from_cid(pubid)
    smile = compound.isomeric_smiles
    smile_list.append(smile)
    inchi = compound.inchi
    inchi_list.append(inchi)

pub_df['smiles'] = smile_list
pub_df['inchi'] = inchi_list

pub_df.to_csv('./GDSC_data/smile_inchi.csv')