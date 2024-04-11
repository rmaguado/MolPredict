import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, SaltRemover
from rdkit import RDLogger
from molvs import Standardizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
import warnings
import copy
import umap

warnings.simplefilter('ignore', UserWarning)

def get_mfp(df_smiles):
    mol_list = df_smiles["smiles"].unique()
    fingerprints, fingerprints1 = [], []
    smiles_list = []
    std_smiles = []
    error_smiles = []
    RDLogger.DisableLog('rdApp.*')
    for m in mol_list:
        try:
            m1 = copy.copy(m)
            m = Chem.MolFromSmiles(m)
            remover = SaltRemover.SaltRemover()  # remove salt
            m = remover.StripMol(m)
            s = Standardizer()  # standardize molecule
            m = s.standardize(m)
            fp1 = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024)
            fingerprints1.append(fp1)
            fingerprints.append(list(fp1))
            smiles_list.append(m1)
            std_smiles.append(Chem.MolToSmiles(m))
        except:
            error_smiles.append(m1)
    col_name = ['MFP_' + str(i) for i in range(1024)]
    df_mfp_aux = pd.DataFrame(data=fingerprints, columns=col_name)
    df_smile_valid = pd.DataFrame(data=smiles_list, columns=['smiles'])
    df_mfp = pd.concat([df_smile_valid, df_mfp_aux], axis=1)
    return df_mfp, fingerprints1

def assign_cluster_id(df_data, cluster_id):
    df_data['Cluster_ID'] = cluster_id.labels_
    return df_data

def umap_clustering_best(sample, df_data, n_clusters=20):
    """
    Best clustering method
    :param sample: MPFs
    :param df_data: NSC, SMILES
    :param n_clusters: Default 7 clusters
    :return: [SMILES, Cluster_ID]
    """
    x_red = umap.UMAP(n_neighbors=100, min_dist=0.0,
                      n_components=2, metric='jaccard',
                      random_state=42).fit_transform(sample)
    clustering = AgglomerativeClustering(linkage='ward', n_clusters=n_clusters)
    clustering.fit(x_red)
    # Assign cluster ID
    df_clusters = assign_cluster_id(df_data, clustering)
    # Metrics
    s1 = silhouette_score(x_red, clustering.labels_, metric='euclidean')
    c1 = calinski_harabasz_score(x_red, clustering.labels_)
    d1 = davies_bouldin_score(x_red, clustering.labels_)
    df_metrics = pd.DataFrame(data=[[s1, c1, d1]],
                              columns=['Silhouette', 'CH score', 'DB score'])
    return df_metrics, df_clusters