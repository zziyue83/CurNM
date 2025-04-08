import itertools
import logging
import os
import pathlib
import random
import sys

import numpy as np
import pandas as pd
# Enforce tgb to use the given directorty instead of the local directory.
from tgb.utils import info
info.PROJ_DIR = './DATA/'

from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset
import torch
from tqdm import tqdm

from utils import set_logger, set_random_seed 


def load_data(name, val_ratio=0.15, test_ratio=0.15, nn_ratio=0.1, dim=128):
    set_random_seed(2020)

    TGL_DATASETS = ['WIKI', 'REDDIT', 'MOOC', 'LASTFM']
    TGB_DATASETS = ['tgbl_wiki', 'tgbl_review', 'tgbl_coin', 'tgbl_comment', 'tgbl_flight']
    DYG_DATASETS = ['CanParl', 'Contacts', 'enron', 'Flights', 'lastfm', 'mooc', 'myket', 'reddit', 'SocialEvo', 'uci', 'UNtrade', 'UNvote', 'USLegis', 'wikipedia']
    # df is a list of edges，columns: ["src", "dst", "time", "int_roll", "ext_roll"]
    # nfeat: torch.Tensor, shape(num_nodes, dim)
    # efeat: torch.Tensor, shape(num_edges, dim)
    if name in TGL_DATASETS:
        logging.info('Loading TGL datasets.')
        PROJ_DIR = './DATA/'
        df, nfeat, efeat = load_data_TGL(name)
    elif name in TGB_DATASETS:
        logging.info('Loading TGB datasets.')
        PROJ_DIR = './DATA/'
        df, nfeat, efeat = load_data_TGB(name)
    elif name in DYG_DATASETS:
        logging.info('Loading DyGLib datasets.')
        PROJ_DIR = './DATA/'
        df, nfeat, efeat = load_data_DyG(name)
    else:
        raise NotImplementedError(name)

    assert len(df) == len(efeat), "The number of edges isn't equal to the dimension of edge features."
    assert np.all(df['time'].values[1:] - df['time'].values[:-1] >= 0), "Edges are not chronologically increasing."
    # Add a column of "eid" for the function `gen_graph`.
    df['eid'] = np.arange(len(df))

    num_nodes, num_edges = len(nfeat), len(efeat)
    # Split df into train_data, val_data, test_data, nn_val_data, nn_test_data according to TGAT and DyGLib.
    ts = df['time'].values.astype(np.float32)
    val_ts, test_ts = np.quantile(ts, (1-val_ratio-test_ratio, 1-test_ratio))
    val_mask = np.logical_and(val_ts <= ts, ts < test_ts)
    test_mask = test_ts <= ts
    val_df, test_df = df[val_mask], df[test_mask]
    # Remove edges in the training set about nn_ratio% of nodes in the test set .
    test_nodes = set(test_df['src'].unique()).union(set(test_df['dst'].unique()))
    nn_test_nodes = set(random.sample(list(test_nodes), int(nn_ratio * num_nodes)))
    observed_mask = np.logical_and(~df['src'].isin(nn_test_nodes), ~df['dst'].isin(nn_test_nodes))
    train_mask = np.logical_and(ts < val_ts, observed_mask)
    train_df = df[train_mask]
    # Combine nn_test_nodes and nodes only shown in the valid and test set.
    all_nodes = set(df['src'].unique()).union(df['dst'].unique())
    train_nodes = set(train_df['src'].unique()).union(train_df['dst'].unique())
    new_nodes = all_nodes - train_nodes
    new_node_mask = np.logical_or(df['src'].isin(new_nodes), df['dst'].isin(new_nodes))
    new_val_mask = np.logical_and(val_mask, new_node_mask)
    new_test_mask = np.logical_and(test_mask, new_node_mask)
    new_val_df, new_test_df = df[new_val_mask], df[new_test_mask]

    val_nodes = set(val_df['src'].unique()).union(set(val_df['dst'].unique()))
    new_val_nodes = set(new_val_df['src'].unique()).union(set(new_val_df['dst'].unique()))
    new_test_nodes = set(new_test_df['src'].unique()).union(set(new_test_df['dst'].unique()))
    # Logging.
    logging.info('Dataset %s has %d nodes, %d edges.', name, num_nodes, num_edges)
    logging.info('Training set: %d nodes, %d edges.', len(train_nodes), len(train_df))
    logging.info('Validation set: %d nodes, %d edges.', len(val_nodes), len(val_df))
    logging.info('Test set: %d nodes, %d edges.', len(test_nodes), len(test_df))
    logging.info('New nodes in validation set: %d nodes, %d edges.', len(new_val_nodes), len(new_val_df))
    logging.info('New nodes in test set: %d nodes, %d edges.', len(new_test_nodes), len(new_test_df))

    DATA_DIR = f'./DATA_GRAPH/{name}/'
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)
    # g is a dict of T-CSR data, keys: ["indptr", "indices", "ts", "eid"]
    train_g_path = f'{DATA_DIR}/train_g.npz'
    full_g_path = f'{DATA_DIR}/full_g.npz'
    if not os.path.exists(full_g_path):
        logging.warning('Generate full_g.npz for Dataset %s.', name)
        train_g = gen_graph(train_df, num_nodes, add_reverse=True)
        full_g = gen_graph(df, num_nodes, add_reverse=True)
        # train_g, full_g = gen_graph(train_df, df)
        np.savez(f'{DATA_DIR}/train_g.npz', **train_g)
        np.savez(f'{DATA_DIR}/full_g.npz', **full_g)
    train_g, full_g = np.load(train_g_path), np.load(full_g_path)

    return train_g, full_g, df, train_df, val_df, test_df, new_val_df, new_test_df, nfeat, efeat

def load_data_TGL(name, val_ratio=0.15, test_ratio=0.15, dim=128):
    TGL_DIR = './DATA/'
    DATA_DIR = f'{TGL_DIR}/{name}/'

    # df is a list of edges，columns: ["src", "dst", "time", "int_roll", "ext_roll"]
    df = pd.read_csv(f'{DATA_DIR}/edges.csv')
    num_edges = len(df)
    num_nodes = max(int(df['src'].max()), int(df['dst'].max())) + 1

    if os.path.exists(f'{DATA_DIR}/node_features.pt'):
        nfeat = torch.load(f'{DATA_DIR}/node_features.pt').float()
    else:
        nfeat = torch.randn(num_nodes, dim)
    
    if os.path.exists(f'{DATA_DIR}/edge_features.pt'):
        efeat = torch.load(f'{DATA_DIR}/edge_features.pt').float()
    else:
        efeat = torch.randn(num_edges, dim)

    return df, nfeat, efeat

def load_data_TGB(name, val_ratio=0.15, test_ratio=0.15, dim=128):
    TGB_DIR = './DATA/'
    DATA_DIR = f'{TGB_DIR}/datasets/{name}/'

    # Download url use `-` to replace `_`.
    _name = name.replace('_', '-')
    info.PROJ_DIR = TGB_DIR
    logging.info(info.PROJ_DIR)
    dataset = PyGLinkPropPredDataset(name=_name, root="datasets").dataset
    # _df: dict, keys: ["sources", "destinations", "timestamps", "edge_idxs", "edge_feat", "w", "edge_label"]
    # nfeat: np.ndarray, shape(num_nodes, dim)
    # efeat: np.ndarray, shape(num_edges, dim)
    _df, nfeat, efeat = dataset.full_data, dataset._node_feat, dataset._edge_feat
    num_edges = len(_df['edge_idxs'])
    num_nodes = max(int(_df['sources'].max()), int(_df['destinations'].max())) + 1
    if nfeat is None:
        nfeat = torch.randn(num_nodes, dim)
    else:
        nfeat = torch.from_numpy(nfeat).float()
    if efeat is None:
        efeat = torch.randn(num_edges, dim)
    else:
        efeat = torch.from_numpy(efeat).float()

    _, val_mask, test_mask = dataset._train_mask, dataset._val_mask, dataset._test_mask
    ext_roll = np.zeros((num_edges,), dtype=int)
    ext_roll[val_mask] = 1
    ext_roll[test_mask] = 2

    # df is a list of edges，columns: ["src", "dst", "time", "int_roll", "ext_roll"]
    df = pd.DataFrame({
        'src': _df['sources'],
        'dst': _df['destinations'],
        'time': _df['timestamps'],
        'int_roll': np.zeros((num_edges,), dtype=int),
        'ext_roll': ext_roll

    })

    return df, nfeat, efeat

def load_data_DyG(name, val_ratio=0.15, test_ratio=0.15, dim=128):
    DyG_DIR = './DATA/'
    DATA_DIR = f'{DyG_DIR}/{name}/'

    # _df: pd.DataFrame, columns: ["u", "i", "ts", "label", "idx"]
    # nfeat: np.ndarray, shape(num_nodes, node_dim)
    # efeat: np.ndarray, shape(num_edges, edge_dim)
    _df = pd.read_csv(f'{DATA_DIR}/ml_{name}.csv')
    nfeat = np.load(f'{DATA_DIR}/ml_{name}_node.npy')
    nfeat = torch.from_numpy(nfeat).float()
    efeat = np.load(f'{DATA_DIR}/ml_{name}.npy')
    # DyGLib labels the edge idx not all from 0.
    efeat = efeat[_df['idx']]
    efeat = torch.from_numpy(efeat).float()
    num_nodes, num_edges = len(nfeat), len(efeat)

    ts = _df['ts'].values.astype(np.float64)
    val_time, test_time = list(np.quantile(ts, [(1 - val_ratio - test_ratio), (1 - test_ratio)]))

    # train_mask = ts <= val_time
    val_mask = np.logical_and(ts <= test_time, ts > val_time)
    test_mask = ts > test_time
    ext_roll = np.zeros((num_edges,), dtype=int)
    ext_roll[val_mask] = 1
    ext_roll[test_mask] = 2

    # df is a list of edges，columns: ["src", "dst", "time", "int_roll", "ext_roll"]
    df = pd.DataFrame({
        'src': _df['u'],
        'dst': _df['i'],
        'time': _df['ts'],
        'int_roll': np.zeros((num_edges,), dtype=int),
        'ext_roll': ext_roll

    })
    
    return df, nfeat, efeat

def gen_graph(df, num_nodes, add_reverse=True):
    indptr = np.zeros(num_nodes + 1, dtype=int)
    indices = [[] for _ in range(num_nodes)]
    ts = [[] for _ in range(num_nodes)]
    eid = [[] for _ in range(num_nodes)]

    for _, row in tqdm(df.iterrows(), total=len(df)):
        idx, src, dst = int(row['eid']), int(row['src']), int(row['dst'])
        indices[src].append(dst)
        ts[src].append(row['time'])
        eid[src].append(idx)
        if add_reverse:
            indices[dst].append(src)
            ts[dst].append(row['time'])
            eid[dst].append(idx)

    for i in tqdm(range(num_nodes)):
        indptr[i + 1] = indptr[i] + len(indices[i])
    
    indices = np.array(list(itertools.chain(*indices)))
    ts = np.array(list(itertools.chain(*ts)))
    eid = np.array(list(itertools.chain(*eid)))

    logging.info('Sorting...')
    def tsort(i, _indptr, _indices, _t, _eid):
        beg = _indptr[i]
        end = _indptr[i + 1]
        sidx = np.argsort(_t[beg:end])
        _indices[beg:end] = _indices[beg:end][sidx]
        _t[beg:end] = _t[beg:end][sidx]
        _eid[beg:end] = _eid[beg:end][sidx]

    for i in tqdm(range(num_nodes)):
        tsort(i, indptr, indices, ts, eid)

    graph = {
        'indptr': indptr,
        'indices': indices,
        'ts': ts,
        'eid': eid
    }

    return graph

def data_stat(name):
    pass

if __name__ == '__main__':
    set_logger(method='data_adptor', config_str='')

    # TGL_DATASETS = ['WIKI', 'REDDIT', 'MOOC', 'LASTFM']
    # for name in TGL_DATASETS:
    #     logging.info(name)
    #     train_g, full_g, df, nfeat, efeat = load_data(name)
    #     logging.info('T-SCR Graph: %s', ','.join([str((k, full_g[k].shape)) for k in full_g]))
    #     logging.info('DataFrame: shape %s columns %s', str(df.shape), str(df.columns))
    #     logging.info('Node features: %s', str(nfeat.shape))
    #     logging.info('Edge features: %s', str(efeat.shape))
    
    # TGB_DATASETS = ['tgbl_wiki', 'tgbl_review', 'tgbl_coin', 'tgbl_comment', 'tgbl_flight']
    # for name in TGB_DATASETS:
    #     logging.info(name)
    #     train_g, full_g, df, nfeat, efeat = load_data(name)
    #     logging.info('T-SCR Graph: %s', ','.join([str((k, full_g[k].shape)) for k in full_g]))
    #     logging.info('DataFrame: shape %s columns %s', str(df.shape), str(df.columns))
    #     logging.info('Node features: %s', str(nfeat.shape))
    #     logging.info('Edge features: %s', str(efeat.shape))
    
    DYG_DATASETS = ['CanParl', 'Contacts', 'enron', 'Flights', 'lastfm', 'mooc', 'myket', 'reddit', 'SocialEvo', 'uci', 'UNtrade', 'UNvote', 'USLegis', 'wikipedia']
    for name in DYG_DATASETS:
        logging.info(name)
        train_g, full_g, df, nfeat, efeat = load_data(name)
        logging.info('T-SCR Graph: %s', ','.join([str((k, full_g[k].shape)) for k in full_g]))
        logging.info('DataFrame: shape %s columns %s', str(df.shape), str(df.columns))
        logging.info('Node features: %s', str(nfeat.shape))
        logging.info('Edge features: %s', str(efeat.shape))
