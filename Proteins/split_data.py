import os
import pandas as pd
import numpy as np
import subprocess
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit import RDLogger
from sklearn.model_selection import train_test_split, GroupKFold, KFold
from tqdm import tqdm
from scipy.cluster.hierarchy import linkage, fcluster

from utils import save

RESULT_PATH = "DataSplits/"
PDBBIND_PATH = "Data/PDBBindv2020/"
MMALIGN_PATH = "./MMalign"

SPLIT_RATIO = 0.8
RANDOM_STATE = 2024
MAX_WORKERS = 10

RDLogger.DisableLog('rdApp.*')


def cluster_by_ligand_similarity(df: pd.DataFrame):
    """
    Clusters ligands based on their structural similarity.
    """

    df['mol'] = df['name'].apply(lambda m: Chem.MolFromMol2File(f"{PDBBIND_PATH}/{m}/{m}_ligand.mol2", removeHs=False))
    df['fp'] = df['mol'].apply(
        lambda mol: AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024) if mol else None
    )
    df.dropna(subset=['fp'], inplace=True)

    # calculate similarity matrix
    n = len(df)
    similarity_matrix = np.zeros((n, n))
    fps = df['fp'].tolist()
    for i in tqdm(range(n), desc="Calculating ligand similarities"):
        for j in range(i, n):
            similarity = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            similarity_matrix[i, j] = similarity_matrix[j, i] = similarity

    # hierarchical clustering
    Z = linkage(1 - similarity_matrix, 'ward')
    cluster_ids = fcluster(Z, t=1.25, criterion='distance')
    df['ligandClusterId'] = cluster_ids

    return df, similarity_matrix


def run_mmalign(pair):
    i, j, name_i, name_j = pair
    command = [MMALIGN_PATH, f"{PDBBIND_PATH}/{name_i}/{name_i}_protein.pdb",
               f"{PDBBIND_PATH}/{name_j}/{name_j}_protein.pdb"]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if result.returncode == 0:
        first_idx = result.stdout.find("TM-score=")
        similarity_score = float(result.stdout[first_idx:first_idx + 50].split()[1])
        return i, j, similarity_score
    else:
        return i, j, None


def cluster_by_protein_similarity(df: pd.DataFrame):
    """
    Clusters proteins based on their structural similarity using MM-align.
    """

    num_proteins = len(df)
    pairs = [(i, j, df.iloc[i]['name'], df.iloc[j]['name']) for i in range(num_proteins) for j in
             range(i + 1, num_proteins)]

    similarity_matrix = np.zeros((num_proteins, num_proteins))

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(
            tqdm(executor.map(run_mmalign, pairs), total=len(pairs), desc="Calculating protein similarities"))

    for i, j, score in results:
        if score is not None:
            similarity_matrix[i, j] = similarity_matrix[j, i] = score

    # clustering
    Z = linkage(1 - similarity_matrix, 'ward')
    cluster_ids = fcluster(Z, t=1.25, criterion='distance')

    df['proteinClusterId'] = cluster_ids

    return df, similarity_matrix


def get_file_names(input_folder):
    exclude_folders = ['index', 'readme']
    all_items = os.listdir(input_folder)
    return [
        item for item in all_items
        if os.path.isdir(os.path.join(input_folder, item)) and item.lower() not in exclude_folders
    ]


def custom_cluster_split(df, test_size=0.1, random_state=None, column_title="ligandClusterId"):
    """
    Custom split logic to maintain cluster integrity, including handling
    single-member clusters. Requires a given clusterId column title.
    """
    unique_clusters = df[column_title].unique()
    np.random.seed(random_state)
    np.random.shuffle(unique_clusters)

    split_index = int(len(unique_clusters) * (1 - test_size))
    train_clusters, test_clusters = unique_clusters[:split_index], unique_clusters[split_index:]

    train_set = df[df[column_title].isin(train_clusters)]
    test_set = df[df[column_title].isin(test_clusters)]

    return train_set, test_set


def save_similarity_matrix(similarity_matrix, names, save_path):
    """
    Saves a similarity matrix as a CSV file with the first row and column serving as identifiers.
    """
    similarity_df = pd.DataFrame(similarity_matrix, index=names, columns=names)
    similarity_df.to_csv(save_path)


def cluster_ligands_and_proteins_simultaneously(df):
    """
    Runs ligand and protein clustering in parallel and combines the cluster IDs.
    """

    ligand_similarity_matrix = None
    protein_similarity_matrix = None

    with ThreadPoolExecutor() as executor:
        future_protein = executor.submit(cluster_by_protein_similarity, df)  # TODO slow af
        future_ligand = executor.submit(cluster_by_ligand_similarity, df)

        for future in as_completed([future_ligand, future_protein]):
            result_df, similarity_matrix = future.result()
            if 'ligandClusterId' in result_df.columns:
                df['ligandClusterId'] = result_df['ligandClusterId']
                ligand_similarity_matrix = similarity_matrix
            if 'proteinClusterId' in result_df.columns:
                df['proteinClusterId'] = result_df['proteinClusterId']
                protein_similarity_matrix = similarity_matrix

    # combine clusterIds
    df['combinedClusterId'] = df.apply(lambda x: f"{x.ligandClusterId}_{x.proteinClusterId}", axis=1)

    return df, ligand_similarity_matrix, protein_similarity_matrix


def main():
    file_names = get_file_names(PDBBIND_PATH)[:]
    df = pd.DataFrame({'name': file_names})

    # process pdbbind data
    df, ligand_similarity_matrix, protein_similarity_matrix = cluster_ligands_and_proteins_simultaneously(df)

    # save similarity matrices
    save_similarity_matrix(ligand_similarity_matrix, file_names, f"{RESULT_PATH}ligand_similarity_matrix.csv")
    save_similarity_matrix(protein_similarity_matrix, file_names, f"{RESULT_PATH}protein_similarity_matrix.csv")

    # random
    kf = KFold(n_splits=10, shuffle=True, random_state=2024)
    for i, (train_index, test_index) in tqdm(enumerate(kf.split(df)), total=10):
        train_val, test_set = df.iloc[train_index], df.iloc[test_index]
        train_set, val_set = train_test_split(train_val, test_size=1 / 9, random_state=2024)
        save(f'{RESULT_PATH}random/fold{i}', train_set, test_set, val_set)

    # ligand
    ligand_kf = GroupKFold(n_splits=10)
    for i, (train_index, test_index) in tqdm(enumerate(ligand_kf.split(df, groups=df['ligandClusterId'])), total=10):
        train_val, test_set = df.iloc[train_index], df.iloc[test_index]
        train_set, val_set = custom_cluster_split(train_val, test_size=1 / 9, random_state=2024,
                                                  column_title="ligandClusterId")
        save(f'{RESULT_PATH}ligand/fold{i}', train_set, test_set, val_set)

    # protein
    protein_kf = GroupKFold(n_splits=10)
    for i, (train_index, test_index) in tqdm(enumerate(protein_kf.split(df, groups=df['proteinClusterId'])),
                                             total=10):
        train_val, test_set = df.iloc[train_index], df.iloc[test_index]
        train_set, val_set = custom_cluster_split(train_val, test_size=1 / 9, random_state=2024,
                                                  column_title="proteinClusterId")
        save(f'{RESULT_PATH}protein/fold{i}', train_set, test_set, val_set)

    # both
    both_kf = GroupKFold(n_splits=10)
    for i, (train_index, test_index) in tqdm(enumerate(both_kf.split(df, groups=df['combinedClusterId'])),
                                             total=10):
        train_val, test_set = df.iloc[train_index], df.iloc[test_index]
        train_set, val_set = custom_cluster_split(train_val, test_size=1 / 9, random_state=2024,
                                                  column_title="combinedClusterId")
        save(f'{RESULT_PATH}both/fold{i}', train_set, test_set, val_set)


if __name__ == "__main__":
    main()
