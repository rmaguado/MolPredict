import os
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import argparse
from tqdm import tqdm

from umap_clustering import umap_clustering_best, get_mfp

parser = argparse.ArgumentParser(description='Data segmentation strategies')
parser.add_argument('--dataset', type=str, required=False,help='gdsc, nci60, combined')
parser.add_argument('--result_folder', type=str, required=True, help='The save path of CV10 data')
args = parser.parse_args()

def save(output_folder, train, test, val):
    folder_path = os.path.join(args.result_folder, output_folder)
    if not os.path.exists(folder_path): 
        os.makedirs(folder_path)
    file_paths = [
        os.path.join(folder_path, f'{item}.csv')\
            for item in ['train', 'test', 'val']
    ]
    train = train.drop_duplicates(ignore_index=True)
    test = test.drop_duplicates(ignore_index=True)
    val = val.drop_duplicates(ignore_index=True)
    train.to_csv(file_paths[0], index=False)
    test.to_csv(file_paths[1], index=False)
    val.to_csv(file_paths[2], index=False)

def cluster_cells(cdr):
    used_cell_lines = cdr['cell_line'].unique()
    cell_info = pd.read_csv('./Data/processed/cell_info.csv')
    cell_info = cell_info[cell_info['cell_type'].isin(used_cell_lines)]

    methylation = pd.read_csv('./Data/processed/methylation.csv',index_col=0)
    genetic = pd.read_csv('./Data/processed/genetic.csv',index_col=0)
    expression = pd.read_csv('./Data/processed/expression.csv',index_col=0)

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

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(omics_data)

    kmeans = KMeans(n_clusters=10, random_state=0)
    kmeans.fit(data_scaled)

    labels = kmeans.labels_
    labels = pd.DataFrame(labels)
    labels.columns = ['cell_label']
    labels['cell_line'] = cell_info['cell_type'].values
    labels.to_csv(os.path.join(args.result_folder,'cell_labels.csv'), index=False)

    CDR_cell_cluster = pd.merge(cdr, labels, on='cell_line')
    cell_clusters = labels['cell_label'].unique()
    return CDR_cell_cluster, cell_clusters

def cluster_drugs(cdr):
    df_mfp, fingerprints = get_mfp(cdr)
    metrics, df_clusters = umap_clustering_best(fingerprints, df_mfp, n_clusters=10)
    drug_clusters = df_clusters[['smiles', 'Cluster_ID']]
    drug_clusters.to_csv(os.path.join(args.result_folder,'drug_clusters.csv'), index=False)
    cluster_ids = drug_clusters['Cluster_ID'].unique()
    return drug_clusters, cluster_ids

def cold_drugs(train_i, test_i, cdr, drug_clusters, drug_cluster_ids):
    train_clusters_drug = drug_cluster_ids[train_i]
    test_clusters_drug = drug_cluster_ids[test_i]

    train_smiles = drug_clusters[drug_clusters['Cluster_ID'].isin(train_clusters_drug)]['smiles']
    test_smiles = drug_clusters[drug_clusters['Cluster_ID'].isin(test_clusters_drug)]['smiles']

    train_set = cdr[cdr['smiles'].isin(train_smiles)]
    test_set = cdr[cdr['smiles'].isin(test_smiles)]

    train_set, val_set = train_test_split(train_set, test_size=1/9, random_state=0)
    return train_set, test_set, val_set

def cold_cells(train_i, test_i, cdr, cdr_cell_cluster, cell_cluster_ids):
    train_clusters_cell = cell_cluster_ids[train_i]
    test_clusters_cell = cell_cluster_ids[test_i]

    train_cells = cdr_cell_cluster[cdr_cell_cluster['cell_label'].isin(train_clusters_cell)]['cell_line']
    test_cells = cdr_cell_cluster[cdr_cell_cluster['cell_label'].isin(test_clusters_cell)]['cell_line']

    train_val = cdr[cdr['cell_line'].isin(train_cells)]
    test_set = cdr[cdr['cell_line'].isin(test_cells)]

    train_set, val_set = train_test_split(train_val, test_size=1/9, random_state=0)
    return train_set, test_set, val_set

if args.dataset == 'gdsc':
    gdsc1 = pd.read_csv('./Data/processed/gdsc1_cdr.csv')
    gdsc2 = pd.read_csv('./Data/processed/gdsc2_cdr.csv')
    CDR = pd.concat([gdsc1, gdsc2], axis=0)
elif args.dataset == 'nci60':
    CDR = pd.read_csv('./Data/processed/nci_cdr.csv')
elif args.dataset == 'combined':
    gdsc1 = pd.read_csv('./Data/processed/gdsc1_cdr.csv')
    gdsc2 = pd.read_csv('./Data/processed/gdsc2_cdr.csv')
    nci = pd.read_csv('./Data/processed/nci_cdr.csv')
    CDR = pd.concat([gdsc1, gdsc2, nci], axis=0)
else:
    raise ValueError('Invalid dataset')
CDR = shuffle(CDR, random_state=0)

kf = KFold(n_splits=10, random_state=0, shuffle=True)

print('Warm start')
for i, (train_index, test_index) in tqdm(enumerate(kf.split(CDR)), total=10):
    train_val, test_set = CDR.iloc[train_index], CDR.iloc[test_index]
    train_set, val_set = train_test_split(train_val, test_size=1/9, random_state=0)
    save(f'CV10/fold{i}', train_set, test_set, val_set)

print('Cold drug')
drug_clusters, drug_cluster_ids = cluster_drugs(CDR)
for i, (train_index, test_index) in tqdm(enumerate(kf.split(drug_cluster_ids)), total=10):
    train_set, test_set, val_set = cold_drugs(
        train_index, test_index, CDR, drug_clusters, drug_cluster_ids
    )
    save(f'CV10_cold_drug/fold{i}', train_set, test_set, val_set)

print('Cold cell')
CDR_cell_cluster, cell_cluster_Ids = cluster_cells(CDR)
for i, (train_index, test_index) in tqdm(enumerate(kf.split(cell_cluster_Ids)), total=10):
    train_set, test_set, val_set = cold_cells(
        train_index, test_index, CDR, CDR_cell_cluster, cell_cluster_Ids
    )
    save(f'CV10_cold_cell/fold{i}', train_set, test_set, val_set)

print('Cold drug and cell')
for i, ((train_drug_index, test_drug_index), (train_cell_index, test_cell_index)) in tqdm(enumerate(
    zip(kf.split(drug_cluster_ids), kf.split(cell_cluster_Ids))), total=10):
    train_set_drug, test_set_drug, val_set_drug = cold_drugs(
        train_drug_index, test_drug_index, CDR, drug_clusters, drug_cluster_ids
    )
    train_set_cell, test_set_cell, val_set_cell = cold_cells(
        train_cell_index, test_cell_index, CDR, CDR_cell_cluster, cell_cluster_Ids
    )

    train_val = pd.merge(train_set_drug, train_set_cell,
                         on=['cell_line', 'panel', 'smiles', 'pIC50'],
                         how='inner')
    test_set = pd.merge(test_set_drug, test_set_cell,
                        on=['cell_line', 'panel', 'smiles', 'pIC50'],
                        how='inner')

    train_set, val_set = train_test_split(train_val, test_size=1/9, random_state=0)
    save(f'CV10_cold_drug_cell/fold{i}', train_set, test_set, val_set)
