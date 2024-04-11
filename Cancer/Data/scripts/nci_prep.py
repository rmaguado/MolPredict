import pandas as pd
import numpy as np
import pubchempy as pcp

def cid_to_smiles(cid):
    try:
        mol = pcp.Compound.from_cid(int(cid))
        if mol:
            smile = mol.isomeric_smiles
            return smile
    except:
        pass
    return float('nan')

nci = pd.read_csv("../raw/NCI60/IC50.csv")
nci = nci.loc[nci["CONCENTRATION_UNIT"] == 'M'][['NSC','PANEL_CODE','CELL_NAME', 'AVERAGE']]
nci = nci.groupby(['NSC', 'CELL_NAME', 'PANEL_CODE']).agg({'AVERAGE': 'mean'}).reset_index()

nci_gdsc_codes = pd.read_csv("../processed/nci_codes.csv")
nci_gdsc_codes = nci_gdsc_codes.rename(columns={"nci": "CELL_NAME", "gdsc":"gdsc_name"})

nsc_smiles = pd.read_csv("../raw/NCI60/nsc_sid_cid.csv")
nsc_smiles = nsc_smiles.dropna(subset='CID')

nci_prep = pd.merge(nci, nci_gdsc_codes, on='CELL_NAME', how='inner')
nci_prep = pd.merge(nci_prep, nsc_smiles, on='NSC', how='inner')
nci_prep = nci_prep.dropna(subset=["gdsc_name"])
nci_prep['AVERAGE'] = nci_prep.loc[:, 'AVERAGE'].apply(lambda x : -x - 6)
nci_prep['AVERAGE'] = nci_prep.loc[:, 'AVERAGE'].clip(lower=-4, upper=4)

cid_smiles = pd.DataFrame({'CID':nci_prep['CID'].unique()})
cid_smiles['smiles'] = cid_smiles['CID'].apply(cid_to_smiles)
cid_smiles = cid_smiles.dropna()

nci_prep_smiles = pd.merge(nci_prep, cid_smiles, on="CID", how="inner")
nci_prep_smiles.rename(columns={
    "AVERAGE": "pIC50",
    "SMILES":"smiles",
    "PANEL_CODE": "panel",
    "gdsc_name": "cell_line"
}, inplace=True)
nci_prep_smiles = nci_prep_smiles[pd.notna(nci_prep_smiles["smiles"])]
nci_final = nci_prep_smiles[["cell_line", "panel", "smiles", "pIC50"]]

nci_final.to_csv("../processed/nci_cdr.csv", index=False)