# python3
# -*- coding:utf-8 -*-

"""
@author:野山羊骑士
@e-mail：thankyoulaojiang@163.com
@file：PycharmProject-PyCharm-Step1_stat.py
@time:2021/8/12 9:50 
"""

import sys
import pandas as pd
import numpy as np

cell_line_file = sys.argv[1]
cell_line_df = pd.read_excel(cell_line_file,sheet_name='Cell line details')
print(cell_line_df.head(5))
print(cell_line_df.shape)


print('#'*50)
print('\t Null value statistics：')
print(cell_line_df.count())

print('#'*50)
print('1\t shared cell lines：{}'.format(cell_line_df['COSMIC identifier'].value_counts().shape[0]))
print('2\t Each cell line has an independent COSMIC id')
print('3\t These cell lines correspond to：{} tumor types'.format(
    cell_line_df['Cancer Type\n(matching TCGA label)'].value_counts().shape[0]))
print('\t The specific amount of cell line data corresponding to each tumor is')
print(cell_line_df['Cancer Type\n(matching TCGA label)'].value_counts())
