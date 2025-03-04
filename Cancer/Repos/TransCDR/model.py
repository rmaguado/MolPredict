# python3
# -*- coding:utf-8 -*-

"""
@author:Xiaoqiong Xia
@e-mail: 
@model: multi-omics model
@time:2021/9/15 16:33 
"""

import os
import numpy as np
import pandas as pd
import json
from sklearn.metrics import mean_squared_error
from lifelines.utils import concordance_index
from scipy.stats import pearsonr,spearmanr
from sklearn.metrics import mean_squared_error
import copy
import time
import pickle
import torch
from torch.utils import data
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import SequentialSampler
from dgllife.utils import mol_to_bigraph, PretrainAtomFeaturizer, PretrainBondFeaturizer
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, SaltRemover
from molvs import Standardizer
from prettytable import PrettyTable
from tqdm import tqdm
from dgllife.model import load_pretrained
from dgl.nn.pytorch.glob import AvgPooling
from transformers import AutoModel
#import sys
#sys.path.append('./code/cross_att_clas_regr')
from model_helper import Encoder
from drug_bert_model import LigandTokenizer, embed_ligand

from transformers import logging
logging.set_verbosity_error()

os.environ["TOKENIZERS_PARALLELISM"] = "true"

# get sequence feats with pre-trained BERT model
def get_sequence_feats(drug_df,**config):
    chem_tokenizer = LigandTokenizer(**config)
    # seyonec/PubChem10M_SMILES_BPE_450k, seyonec/ChemBERTa-zinc-base-v1
    lig_embedder = AutoModel.from_pretrained(config['seq_model'])
    lig_embedder.eval()
    embedded_drug = {}
    smiles_list = pd.Series(drug_df['smiles'].unique())
    print("Getting llm features...")
    for smiles in tqdm(smiles_list):
        embedded_drug[smiles] = embed_ligand(smiles, chem_tokenizer, lig_embedder)
    print("Done.")
    return embedded_drug


# get graph feats with pre-trained GIN model
def get_graph_feats(drug_df,**config):
    device = torch.device(config["device"])
    model_drug = load_pretrained(config['graph_model'])
    readout = AvgPooling()
    model_drug.eval()
    model_drug.to(device)
    embedded_drug = {}
    smiles_list = pd.Series(drug_df['smiles'].unique())
    print("Getting graph features...")
    for smiles in tqdm(smiles_list):
        v_D = mol_to_bigraph(Chem.MolFromSmiles(smiles), add_self_loop=True,
                                                    node_featurizer=PretrainAtomFeaturizer(),
                                                    edge_featurizer=PretrainBondFeaturizer(),
                                                    canonical_atom_order=False)
        v_D = v_D.to(device)
        nfeats = [v_D.ndata.pop('atomic_number').to(device),
                v_D.ndata.pop('chirality_type').to(device)]
        efeats = [v_D.edata.pop('bond_type').to(device),
                v_D.edata.pop('bond_direction_type').to(device)]
        node_repr = model_drug(v_D, nfeats, efeats)
        v_D = readout(v_D, node_repr)
        embedded_drug[smiles] = v_D.detach().cpu().numpy().flatten()
    print("Done.")
    return embedded_drug

# get ECFP feats
def smiles2morgan(smiles, radius=2, nBits=1024):
    try:
        mol = Chem.MolFromSmiles(smiles)
        features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, features)
    except:
        print('rdkit not found this smiles for morgan: ' + smiles + ' convert to all 0 features')
        features = np.zeros((nBits, ))
    return features

def get_FP_feats(drug_df,**config):
    print("Getting fingerprints...")
    embedded_drug = {}
    smiles_list = pd.Series(drug_df['smiles'].unique())
    for smiles in tqdm(smiles_list):
        embedded_drug[smiles] = smiles2morgan(smiles, radius = 2, nBits = 1024)
    print("Done.")
    return embedded_drug


# CNN & RNN
smiles_char = ['?', '#', '%', ')', '(', '+', '-', '.', '1', '0', '3', '2', '5', '4',
       '7', '6', '9', '8', '=', 'A', 'C', 'B', 'E', 'D', 'G', 'F', 'I',
       'H', 'K', 'M', 'L', 'O', 'N', 'P', 'S', 'R', 'U', 'T', 'W', 'V',
       'Y', '[', 'Z', ']', '_', 'a', 'c', 'b', 'e', 'd', 'g', 'f', 'i',
       'h', 'm', 'l', 'o', 'n', 's', 'r', 'u', 't', 'y']

def trans_drug(x):
    MAX_SEQ_DRUG = 100
    temp = list(x)
    temp = [i if i in smiles_char else '?' for i in temp]
    if len(temp) < MAX_SEQ_DRUG:
        temp = temp + ['?'] * (MAX_SEQ_DRUG-len(temp))
    else:
        temp = temp [:MAX_SEQ_DRUG]
    return temp

def drug_2_embed(drug_df):
    from sklearn.preprocessing import OneHotEncoder
    enc_drug = OneHotEncoder().fit(np.array(smiles_char).reshape(-1, 1))
    smiles_list = pd.Series(drug_df['smiles'].unique())
    embedded_drug = {}
    for smiles in tqdm(smiles_list):
        smiles_split = trans_drug(smiles)
        embedded_drug[smiles] = enc_drug.transform(np.array(smiles_split).reshape(-1,1)).toarray().T
    return embedded_drug


# transformer
from subword_nmt.apply_bpe import BPE # Byte-Pair-Encoder
import codecs
vocab_path = './ESPF/drug_codes_chembl_freq_1500.txt'
bpe_codes_drug = codecs.open(vocab_path)
dbpe = BPE(bpe_codes_drug, merges=-1, separator='')
sub_csv = pd.read_csv('./ESPF/subword_units_map_chembl_freq_1500.csv')
idx2word_d = sub_csv['index'].values
words2idx_d = dict(zip(idx2word_d, range(0, len(idx2word_d))))

def drug2emb_encoder(x):
    max_d = 50
    t1 = dbpe.process_line(x).split()  # split
    try:
        i1 = np.asarray([words2idx_d[i] for i in t1])  # index
    except:
        i1 = np.array([0])
    l = len(i1)
    if l < max_d:
        i = np.pad(i1, (0, max_d - l), 'constant', constant_values = 0)
        input_mask = ([1] * l) + ([0] * (max_d - l))
    else:
        i = i1[:max_d]
        input_mask = [1] * max_d
    return i, np.asarray(input_mask)

def get_transformer_feats(drug_df):
    smiles_list = pd.Series(drug_df['smiles'].unique())
    embedded_drug = {}
    mask = {}
    for smiles in tqdm(smiles_list):
        embedded_drug[smiles],mask[smiles] = drug2emb_encoder(smiles)
    return embedded_drug,mask


# GCN and NeuralFP
def get_GNN_feats(drug_df,**config):
    from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer,AttentiveFPAtomFeaturizer,AttentiveFPBondFeaturizer
    from functools import partial
    smiles_list = pd.Series(drug_df['smiles'].unique())
    embedded_drug = {}
    for smiles in tqdm(smiles_list): 
        embedded_drug[smiles] = fc(smiles = smiles, node_featurizer = node_featurizer, edge_featurizer = edge_featurizer)
    return embedded_drug


class data_process_loader(data.Dataset):
    def __init__(self, list_IDs, labels, drug_df,**config):
        'Initialization'
        self.config = config
        self.labels = labels
        self.list_IDs = list_IDs
        self.drug_df = drug_df
        
        self.rna_data = pd.read_csv('../../Data/processed/methylation.csv', index_col=0)
        self.rna_data.columns = [name.split('.')[0][1:] for name in self.rna_data.columns.values]
        self.rna_data = self.rna_data.T
        self.genetic = pd.read_csv('../../Data/processed/genetic.csv',index_col=0)
        self.mrna = pd.read_csv('../../Data/processed/expression.csv',index_col=0)
        self.mrna = self.mrna.T
        
        self.drug_encodings = self.config['drug_model'].split(' + ')

        if 'sequence' in self.drug_encodings:
            self.embedded_drug1 = get_sequence_feats(drug_df, **self.config)
        if 'graph' in self.drug_encodings:
            self.embedded_drug2 = get_graph_feats(drug_df, **self.config)
        if 'FP' in self.drug_encodings:
            self.embedded_drug3 = get_FP_feats(drug_df, **self.config)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        index = self.list_IDs[index]
        y = self.labels[index]
        
        v_rna = np.array(self.rna_data.loc[self.drug_df.iloc[index]['assay_name'],:])
        v_genetic = np.array(self.genetic.loc[int(self.drug_df.iloc[index]['COSMIC_ID']),:])
        v_mrna = np.array(self.mrna.loc[self.drug_df.iloc[index]['cell_line'],:])
    
        v_d = []
        if 'sequence' in self.drug_encodings:
            v_d.append(self.embedded_drug1[self.drug_df.iloc[index]['smiles']])
        if 'graph' in self.drug_encodings:
            v_d.append(self.embedded_drug2[self.drug_df.iloc[index]['smiles']])
        if 'FP' in self.drug_encodings:
            v_d.append(self.embedded_drug3[self.drug_df.iloc[index]['smiles']])
        return tuple(v_d), v_rna, v_genetic, v_mrna, y

class MLP(nn.Sequential):
    def __init__(self,input_dim_gene, device):
        self.input_dim_gene = input_dim_gene
        self.device = device
        hidden_dim_gene = 256
        mlp_hidden_dims_gene = [1024, 512]
        super(MLP, self).__init__()
        layer_size = len(mlp_hidden_dims_gene) + 1
        dims = [self.input_dim_gene] + mlp_hidden_dims_gene + [hidden_dim_gene]
        self.predictor = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]).to(device) for i in range(layer_size)])
    def forward(self, v):
        # predict
        v = v.float().to(self.device)
        for i, l in enumerate(self.predictor):
            v = F.relu(l(v))
        return v


class Classifier(nn.Sequential):
    def __init__(self, **config):
        super(Classifier, self).__init__()
        self.config = config
        self.device = torch.device(config["device"])

        # cell line model
        self.model_rna = MLP(self.config['input_dim_rna'], self.device)
        self.model_genetic = MLP(self.config['input_dim_genetic'], self.device)
        self.model_mrna = MLP(self.config['input_dim_mrna'], self.device)

        self.drug_encodings = self.config['drug_model'].split(' + ')
        # pretrained models
        self.drug_models = []
        if 'sequence' in self.drug_encodings:
            self.drug_models.append(MLP(768, self.device))
        if 'graph' in self.drug_encodings:
            self.drug_models.append(MLP(300, self.device))
        if 'FP' in self.drug_encodings:
            self.drug_models.append(MLP(1024, self.device))

        self.hidden_dims =  [1024, 1024, 512]
        
        n_drug = len(self.config['drug_model'].split(' + '))
        n_omics = len(self.config['omics'].split(' + '))
        n = n_drug + n_omics
        dims = [256*n] + self.hidden_dims + [1]

        self.fusion = Encoder(256, 256, 8, 6, 0.1, self.device)
        
        self.dropout = nn.Dropout(0.1)
        layer_size = len(self.hidden_dims) + 1
        self.predictor = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(layer_size)])

    def forward(self, v):
        # label
        label = v[-1]

        # cell line
        v_rna = self.model_rna(v[1])
        v_genetic = self.model_genetic(v[2])
        v_mrna = self.model_mrna(v[3])

        # unsqueeze
        v_rna = v_rna.unsqueeze(1)
        v_genetic = v_genetic.unsqueeze(1)
        v_mrna = v_mrna.unsqueeze(1)

        v_cell = torch.cat((v_rna, v_genetic, v_mrna), 1)

        v_D = torch.cat([model(v_).unsqueeze(1) for model, v_ in zip(self.drug_models, v[0])], dim=1)

        v_f = torch.cat((v_D, v_cell), 1)
        v_f = self.fusion(v_f, None)
        v_f = v_f.view(-1, v_f.shape[1] * v_f.shape[2])

        for i, l in enumerate(self.predictor):
            if i == (len(self.predictor) - 1):
                v_f = l(v_f)
            else:
                v_f = F.relu(self.dropout(l(v_f)))
        return v_f, label

def dgl_collate_func(x):
    d, v1,v2,v3, y = zip(*x)
    import dgl
    d = dgl.batch(d)
    return d, torch.tensor(v1), torch.tensor(v2),torch.tensor(v3), torch.tensor(y)


class TransCDR:
    def __init__(self,**config):
        #model_drug = transformer()
        self.config = config
        self.model = Classifier(**self.config)
        self.modeldir = config['modeldir']
        if not os.path.exists(self.modeldir):
            os.makedirs(self.modeldir)
        self.device = torch.device(self.config['device'])
        self.pkl_file = os.path.join(self.modeldir, "loss_curve_iter.pkl")

    def test(self,datagenerator,model):
        y_label = []
        y_pred = []
        model.eval()
        for i, v in tqdm(enumerate(datagenerator), total=len(datagenerator)):
            score, label = model(v)
            loss_fct = torch.nn.MSELoss()
            n = torch.squeeze(score, 1)
            loss = loss_fct(n, Variable(torch.from_numpy(np.array(label)).float()).to(self.device))
            logits = torch.squeeze(score).detach().cpu().numpy()

            label_ids = label.to('cpu').numpy()
            y_label = y_label + label_ids.flatten().tolist()
            y_pred = y_pred + logits.flatten().tolist()
            outputs = np.asarray([1 if i else 0 for i in (np.asarray(y_pred) >= 0.5)])

        model.train()

        
        return y_label, y_pred, \
            mean_squared_error(y_label, y_pred), \
            np.sqrt(mean_squared_error(y_label, y_pred)), \
            pearsonr(y_label, y_pred)[0], \
            pearsonr(y_label, y_pred)[1], \
            spearmanr(y_label, y_pred)[0], \
            spearmanr(y_label, y_pred)[1], \
            concordance_index(y_label, y_pred), \
            loss


    def train(self, train_drug, test_drug, val_drug):
        lr = self.config['lr']
        decay = self.config['decay']
        batch_size = self.config['batch_size']
        train_epoch = self.config['train_epoch']
        self.model = self.model.to(self.device)
        # self.model = torch.nn.DataParallel(self.model, device_ids=[0, 5])
        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr, weight_decay=decay)
        loss_history = []
        
        print('******** data preparing ********')
        params = {'batch_size': batch_size,
                'shuffle': True,
                'num_workers': 0,
                'drop_last': False,
                }

        train_loader = data_process_loader(
            train_drug.index.values, train_drug.Label.values, train_drug,
        **self.config)
        training_generator = data.DataLoader(train_loader, **params)

        if test_drug is not None:
            test_loader = data_process_loader(
                test_drug.index.values, test_drug.Label.values, test_drug,
            **self.config)
            testing_generator = data.DataLoader(test_loader, **params)
        if val_drug is not None:
            val_loader = data_process_loader(
                val_drug.index.values, val_drug.Label.values, val_drug,
            **self.config)
            validation_generator = data.DataLoader(val_loader, **params)

        valid_metric_record = []
        max_MSE = 10000
        valid_metric_header = ['# epoch',"MSE", 'RMSE',
                            "Pearson Correlation", "with p-value",
                            'Spearman Correlation',"with p-value2",
                            "Concordance Index"]
        
        model_max = copy.deepcopy(self.model)
        table = PrettyTable(valid_metric_header)
        float2str = lambda x: '%0.4f' % x
        
        print('******** Go for Training ********')
        writer = SummaryWriter(self.modeldir, comment='Drug_Transformer_MLP')
        t_start = time.time()
        iteration_loss = 0

        for epo in range(train_epoch):
            for i, v in enumerate(training_generator):
                score, label = self.model(v)
                label = Variable(torch.from_numpy(np.array(label))).float().to(self.device)
                
                loss_fct = torch.nn.MSELoss()
                n = torch.squeeze(score, 1).float()
                loss = loss_fct(n, label)

                loss_history.append(loss.item())
                writer.add_scalar("Loss/train", loss.item(), iteration_loss)
                iteration_loss += 1

                opt.zero_grad()
                loss.backward()
                opt.step()
                if (i % 100 == 0):
                    t_now = time.time()
                    print('Training at Epoch ' + str(epo + 1) +
                          ' iteration ' + str(i) + \
                          ' with loss ' + str(loss.cpu().detach().numpy())[:7] + \
                          ". Total time " + str(int(t_now - t_start) / 3600)[:7] + " hours")
            if val_drug is not None:
                with torch.set_grad_enabled(False):
                
                    ### regression: MSE, Pearson Correlation, with p-value, Concordance Index
                    y_true,y_pred, mse, rmse, \
                    person, p_val, \
                    spearman, s_p_val, CI,\
                    loss_val = self.test(validation_generator, self.model)
                    lst = ["epoch " + str(epo)] + list(map(float2str, [mse, rmse, person, p_val, spearman,
                                                                    s_p_val, CI]))
                    valid_metric_record.append(lst)
                    print('Validation at Epoch ' + str(epo + 1) +
                            ' with loss:' + str(loss_val.item())[:7] +
                            ', MSE: ' + str(mse)[:7] +
                            ' , Pearson Correlation: ' + str(person)[:7] +
                            ' with p-value: ' + str(p_val)[:7] +
                            ' Spearman Correlation: ' + str(spearman)[:7] +
                            ' with p_value: ' + str(s_p_val)[:7] +
                            ' , Concordance Index: ' + str(CI)[:7])
                    writer.add_scalar("valid/mse", mse, epo)
                    writer.add_scalar('valida/rmse', rmse, epo)
                    writer.add_scalar("valid/pearson_correlation", person, epo)
                    writer.add_scalar("valid/concordance_index", CI, epo)
                    writer.add_scalar("valid/Spearman", spearman, epo)
                    writer.add_scalar("Loss/valid", loss_val.item(), iteration_loss)
                    if mse < max_MSE:
                        model_max = copy.deepcopy(self.model)
                        max_MSE = mse
                        es = 0
                    else:
                        es += 1
                        print("Counter {} of 5".format(es))
                        if es > 4:
                            print("Early stopping with best_mse: ", str(max_MSE)[:7], "and mse for this epoch: ", str(mse)[:7], "...")
                            break

                table.add_row(lst)
            else:
                model_max = copy.deepcopy(self.model)
        # load early stopped model
        self.model = model_max
        
        print('******** predicting... ********')
        # make predictions for train
        train_predictions, _ = self.predict(train_drug, train_loader)
        train_predictions.to_csv(os.path.join(self.modeldir, "train_predictions.csv"), index=False)
        
        # make predictions for test
        if test_drug is not None:
            test_predictions, _ = self.predict(test_drug, test_loader)
            test_predictions.to_csv(os.path.join(self.modeldir, "test_predictions.csv"), index=False)
        
        # make predictions for val
        if val_drug is not None:
            val_predictions, _ = self.predict(val_drug, val_loader)
            val_predictions.to_csv(os.path.join(self.modeldir, "val_predictions.csv"), index=False)

            prettytable_file = os.path.join(self.modeldir, "valid_markdowntable.txt")
            with open(prettytable_file, 'w') as fp:
                fp.write(table.get_string())
        
        pkl_file = os.path.join(self.modeldir, "loss_curve_iter.pkl")
        with open(pkl_file, 'wb') as pck:
            pickle.dump(loss_history, pck)
        
        print('******** Go for Testing ********')
        if test_drug is not None:
            y_true,y_pred, mse, rmse, \
            person, p_val, \
            spearman, s_p_val, CI,\
            loss_test = self.test(testing_generator, model_max)
            test_table = PrettyTable(["MSE","RMSE", "Pearson Correlation", "p-value", "spearman","s_p-value","Concordance Index"])
            test_table.add_row(list(map(float2str, [mse,rmse, person, p_val,spearman, s_p_val, CI])))
            print('Testing MSE: ' + str(mse) 
                + ' , Pearson Correlation: ' + str(person) + ' with p-value: ' + str(f"{p_val:.2E}") 
                + ' , Spearman Correlation: ' + str(spearman) + ' with s_p-value: ' + str(f"{s_p_val:.2E}")
                + ' , Concordance Index: '+str(CI))

            np.save(os.path.join(self.modeldir, "logits.npy"), np.array(y_pred))
            # att.to_csv(os.path.join(self.modeldir, "attention.csv"))
            prettytable_file = os.path.join(self.modeldir, "test_markdowntable.txt")
            with open(prettytable_file, 'w') as fp:
                fp.write(test_table.get_string())

        print('******** Training Finished ********')
        writer.flush()
        writer.close()

        tf = open(os.path.join(self.modeldir, "config.json"), "w")
        json.dump(self.config,tf)
        tf.close()

    def predict(self, drug_data, data_loader):
        prediction_data = drug_data.copy()
        self.model.to(self.device)
        params = {'batch_size': self.config['batch_size'],
                  'shuffle': False,
                  'num_workers': 8,
                  'drop_last': False,
                  'sampler': SequentialSampler(data_loader)}
        generator = data.DataLoader(data_loader, **params)
        y_label, y_pred, mse, rmse, person, p_val, spearman, s_p_val, CI, loss_val = \
            self.test(generator, self.model)
        
        y_label = np.array(y_label)
        y_pred = np.array(y_pred)
        
        prediction_data['Prediction'] = y_pred
        prediction_data['Residual'] = y_label - y_pred
        columns = ["cell_line", "smiles", "Label", "Prediction", "Residual"]
        prediction_data = prediction_data[columns]
        return prediction_data, (mse, rmse, person, p_val, spearman, s_p_val, CI)

    def save_model(self):
        torch.save(self.model.state_dict(), self.modeldir + '/model.pt')

    def load_pretrained(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

        if 'cuda' in self.condig["device"]:
            state_dict = torch.load(path)
        else:
            state_dict = torch.load(path, map_location=torch.device('cpu'))

        if next(iter(state_dict))[:7] == 'module.':
            # the pretrained model is from data-parallel module
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            state_dict = new_state_dict

        self.model.load_state_dict(state_dict)
