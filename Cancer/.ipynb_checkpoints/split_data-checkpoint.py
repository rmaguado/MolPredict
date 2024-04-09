import os
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from rdkit.Chem.Scaffolds import MurckoScaffold
import argparse

from umap_clustering import umap_clustering_best, get_mfp

parser = argparse.ArgumentParser(description='Data segmentation strategies')
parser.add_argument('--dataset', type=str, required=False,help='gdsc, nci60, combined')
parser.add_argument('--result_folder', type=str, required=True, help='The save path of CV10 data')
args = parser.parse_args()

def generate_scaffold(smiles, include_chirality=False):
    """
    Obtain assert from smiles
    :param smiles:
    :param include_chirality:
    :return: smiles of scaffold
    """
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        smiles=smiles, includeChirality=include_chirality)
    return scaffold

def save(output_folder, train, test, val):
    folder_path = os.path.join(args.result_folder, output_folder)
    if not os.path.exists(folder_path): 
        os.makedirs(folder_path)
    file_paths = [
        os.path.join(folder_path, f'{item}.csv')\
            for item in ['train', 'test', 'val']
    ]
    train.to_csv(file_paths[0],index=False)
    test.to_csv(file_paths[1],index=False)
    val.to_csv(file_paths[2],index=False)

def load(source_folder):
    files_paths = [
        os.path.join(args.result_folder, source_folder, f'{item}.csv')\
            for item in ['train', 'test', 'val']
    ]
    train = pd.read_csv(files_paths[0], index_col=False)
    test = pd.read_csv(files_paths[1], index_col=False)
    val = pd.read_csv(files_paths[2], index_col=False)
    return train, test, val

def merge_with(train, test, val, other, on):
    train = pd.merge(train, other, on=on)
    test = pd.merge(test, other, on=on)
    val = pd.merge(val, other, on=on)
    return train, test, val

CDR = pd.read_csv('./data/gdsc1_cdr_preprocess.csv')
CDR = shuffle(CDR, random_state=0)

cell_info = pd.read_csv('./data/cell_info.csv')
methylation = pd.read_csv('./data/methylation.csv',index_col=0)
genetic = pd.read_csv('./data/genetic.txt',sep='\t',index_col=0)
expression = pd.read_csv('./data/expression.csv',index_col=0)

methylation.columns = [name.split('.')[0][1:] for name in methylation.columns.values]
methylation = methylation.T
expression = expression.T

methylation = methylation.loc[cell_info['assay_name']]
genetic = genetic.loc[cell_info['COSMIC_ID']]
expression = expression.loc[cell_info['cell_type']]

methylation = methylation.reset_index(drop=True)
genetic = genetic.reset_index(drop=True)
expression = expression.reset_index(drop=True)

omics_data = pd.concat([methylation,genetic,expression],axis=1)
omics_data.index = cell_info['cell_type']

kf = KFold(n_splits=10, random_state=0, shuffle=True)

##### warm start #####
for i, (train_index, test_index) in enumerate(kf.split(CDR)):
    train_val, test_set = CDR.iloc[train_index], CDR.iloc[test_index]
    train_set, val_set = train_test_split(train_val, test_size=1/9, random_state=0)
    save(f'CV10/fold{i}', train_set, test_set, val_set)

##### cold drug #####
df_mfp, fingerprints = get_mfp(CDR)
df_metrics, df_clusters = umap_clustering_best(fingerprints, df_mfp, n_clusters=10)
unique_clusters = df_clusters['Cluster_ID'].unique()
for i, (train_index, test_index) in enumerate(kf.split(unique_clusters)):
    train_clusters = unique_clusters[train_index]
    test_clusters = unique_clusters[test_index]

    train_smiles = df_clusters[df_clusters['Cluster_ID'].isin(train_clusters)]['smiles']
    test_smiles = df_clusters[df_clusters['Cluster_ID'].isin(test_clusters)]['smiles']

    train_set = CDR[CDR['smiles'].isin(train_smiles)]
    test_set = CDR[CDR['smiles'].isin(test_smiles)]

    train_set, val_set = train_test_split(train_set, test_size=1/9, random_state=0)
    save(f'CV10_cold_drug/fold{i}', train_set, test_set, val_set)

##### cold cell #####
scaler = StandardScaler()
data_scaled = scaler.fit_transform(omics_data)

kmeans = KMeans(n_clusters=10, random_state=0)
kmeans.fit(data_scaled)

labels = kmeans.labels_
labels = pd.DataFrame(labels)
labels.columns = ['cell_label']
labels['cell_line'] = cell_info['cell_type'].values

CDR_cell_cluster = pd.merge(CDR, labels, on='cell_line')
label = labels.drop_duplicates(subset=['cell_label'])
for i, (train_index, test_index) in enumerate(kf.split(label)):
    train_val, test_set = label.iloc[train_index], label.iloc[test_index]
    train_set, val_set = train_test_split(train_val, test_size=1/9, random_state=0)
    train_cell_label = pd.merge(labels['cell_label'], train_set, on='cell_label')
    test_cell_label = pd.merge(labels['cell_label'], test_set, on='cell_label')
    val_cell_label = pd.merge(labels['cell_label'], val_set, on='cell_label')
    train_set = pd.merge(train_set['cell_label'], CDR_cell_cluster, on='cell_label')
    val_set = pd.merge(val_set['cell_label'], CDR_cell_cluster, on='cell_label')
    test_set = pd.merge(test_set['cell_label'], CDR_cell_cluster, on='cell_label')
    save(f'CV10_cold_cell/fold{i}', train_set, test_set, val_set)

##### cold drug and cell #####
for i, (train_index, test_index) in enumerate(kf.split(unique_clusters)):
    train_clusters = unique_clusters[train_index]
    test_clusters = unique_clusters[test_index]

    train_smiles = df_clusters[df_clusters['Cluster_ID'].isin(train_clusters)]['smiles']
    test_smiles = df_clusters[df_clusters['Cluster_ID'].isin(test_clusters)]['smiles']

    train_set_drug = CDR[CDR['smiles'].isin(train_smiles)]
    test_set_drug = CDR[CDR['smiles'].isin(test_smiles)]

    train_set_cell = CDR_cell_cluster[CDR_cell_cluster['cell_label'].isin(train_clusters)]
    test_set_cell = CDR_cell_cluster[CDR_cell_cluster['cell_label'].isin(test_clusters)]

    train_set = pd.merge(train_set_drug, train_set_cell, on=['cell_line', 'smiles'])
    test_set = pd.merge(test_set_drug, test_set_cell, on=['cell_line', 'smiles'])

    train_set, val_set = train_test_split(train_set, test_size=1/9, random_state=0)
    save(f'CV10_cold_drug_cell/fold{i}', train_set, test_set, val_set)
