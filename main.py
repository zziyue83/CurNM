import argparse
import os
import logging
import socket

import torch
import time
import random
import dgl
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from scipy.special import softmax
from scipy.sparse import coo_matrix
from tqdm import tqdm

from data_adaptor import load_data
from memorys import MailBox
from sampler import NegLinkSampler, NegLinkInductiveSampler
from sampler_core import ParallelSampler, TemporalGraphBlock
from utils import EarlyStopMonitor, NegativeEdgeSampler, parse_config, to_dgl_blocks, node_to_dgl_blocks, mfgs_to_cuda, prepare_input, get_ids, get_pinned_buffers, set_logger, set_random_seed, get_neighbor_sampler
from t2v_dens.modules import *
from Ring_dens import ring_robust, policy

parser=argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='dataset name')
parser.add_argument('--seed', type=int, default=0, help='set random seed')
parser.add_argument('--config', type=str, help='path to config file')
parser.add_argument('--gpu', type=str, default='0', help='which GPU to use')
parser.add_argument('--model_name', type=str, default='', help='name of stored model')
parser.add_argument('--use_inductive', action='store_true')
parser.add_argument('--train_neg', type=str, default='CurNM', help='use different negative sampling for training')
parser.add_argument('--rand_edge_features', type=int, default=0, help='use random edge features')
parser.add_argument('--rand_node_features', type=int, default=64, help='use random node features')
parser.add_argument('--eval_neg_samples', type=int, default=1, help='how many negative samples to use at inference. Note: this will change the metric of test set to AP+AUC to AP+MRR!')
parser.add_argument('--lbd', type=float, default=0.3)
parser.add_argument('--mu', type=float, default=1e-13)
parser.add_argument('--M', type=int, default=8)
parser.add_argument('--sample_neighbor_strategy', type=str, default='time_interval_aware', choices=['uniform', 'recent', 'time_interval_aware'], help='how to sample historical neighbors')
parser.add_argument('--time_scaling_factor', default=1e-6, type=float, help='the hyperparameter that controls the sampling preference with time interval, '
                    'a large time_scaling_factor tends to sample more on recent links, 0.0 corresponds to uniform sampling, '
                    'it works when sample_neighbor_strategy == time_interval_aware')
parser.add_argument('--anneal_mode', type=str, default="adaptive_anneal_on_loss")
parser.add_argument('--max_anneal_epoch', type=int, default=100)
parser.add_argument('--t2v_hiddem_dim', type=int, default=32)
parser.add_argument('--device', type=str)
parser.add_argument('--unique', action='store_false', default=True)
parser.add_argument('--add_random', action='store_false', default=True)
parser.add_argument('--fixed_update', action='store_true', default=False)
parser.add_argument('--thres_outer_policy')
parser.add_argument('--thres_inner_policy')
parser.add_argument('--thres_inner', type=float, default=0)
parser.add_argument('--thres_outer', type=float, default=0.1)
parser.add_argument('--global_step', type=int, default=0)
parser.add_argument('--num_epoch', type=int)
parser.add_argument('--max_hist_prop', type=float, default=0.6)
parser.add_argument('--stop_irr_epoch', type=float, default=10)
parser.add_argument('--std_weight', type=float, default=0.006)
parser.add_argument('--cache_threshold', type=float, default=0.5)
parser.add_argument('--min_rand_prop', type=float, default=0.3)
parser.add_argument('--shrinkage', type=float, default=0.03)
args=parser.parse_args()
args.hostname = socket.gethostname()

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID' 
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
os.environ['NUMBA_NUM_THREADS'] = '8'

set_logger(f'{args.data}-{args.model_name}', args.train_neg, log_file=True)
set_random_seed(args.seed)

logging.info(__file__)
logging.info(args)
logging.info('Loading Dataset %s.', args.data)
train_g, full_g, df, train_df, val_df, test_df, new_val_df, new_test_df, nfeat, efeat = load_data(args.data)
df, train_df, val_df, test_df, new_val_df, new_test_df = [tmp.reset_index(drop=True) for tmp in [df, train_df, val_df, test_df, new_val_df, new_test_df]]
sample_param, memory_param, gnn_param, train_param = parse_config(args.config)
df['eid'] = np.arange(len(df))

logging.info('Sample param: %s.', str(sample_param))
logging.info('Memory param: %s.', str(memory_param))
logging.info('GNN param: %s.', str(gnn_param))
logging.info('Train param: %s.', str(train_param))

args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_nodes, num_edges = len(nfeat), len(efeat)
if 'dim_out' in gnn_param:
    dim_out = gnn_param['dim_out']
else:
    dim_out = memory_param['dim_out']
args.num_epoch = train_param['epoch']
batch_size = train_param['batch_size']
num_batch = int(np.ceil(len(train_df) / batch_size))
chunk_size = train_param['reorder'] if 'reorder' in train_param else 16

logging.info('Config Model %s.', args.model_name)
gnn_dim_node, gnn_dim_edge = nfeat.shape[1], efeat.shape[1]
combine_first = False
if 'combine_neighs' in train_param and train_param['combine_neighs']:
    combine_first = True
model = GeneralModel(gnn_dim_node, gnn_dim_edge, sample_param, memory_param, gnn_param, train_param, combined=combine_first, args=args).cuda()
mailbox = MailBox(memory_param, num_nodes, gnn_dim_edge) if memory_param['type'] != 'none' else None
creterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=train_param['lr'])
if 'all_on_gpu' in train_param and train_param['all_on_gpu']:
    if nfeat is not None:
        nfeat = nfeat.cuda()
    if efeat is not None:
        efeat = efeat.cuda()
    if mailbox is not None:
        mailbox.move_to_gpu()

if not ('no_sample' in sample_param and sample_param['no_sample']):
    train_ngh_sampler = ParallelSampler(train_g['indptr'], train_g['indices'], train_g['eid'], train_g['ts'].astype(np.float32),
                                sample_param['num_thread'], 1, sample_param['layer'], sample_param['neighbor'],
                                sample_param['strategy']=='recent', sample_param['prop_time'],
                                sample_param['history'], float(sample_param['duration']))
    full_ngh_sampler = ParallelSampler(full_g['indptr'], full_g['indices'], full_g['eid'], full_g['ts'].astype(np.float32),
                                sample_param['num_thread'], 1, sample_param['layer'], sample_param['neighbor'],
                                sample_param['strategy']=='recent', sample_param['prop_time'],
                                sample_param['history'], float(sample_param['duration']))
else:
    train_ngh_sampler, full_ngh_sampler = None, None
# The same as DyGLib https://github.com/yule-BUAA/DyGLib/train_link_prediction.py.
# initialize negative samplers, set seeds for validation and testing so negatives are the same across different runs
# in the inductive setting, negatives are sampled only amongst other new nodes
# train negative edge sampler does not need to specify the seed, but evaluation samplers need to do so
train_neg_sampler = NegativeEdgeSampler(src_node_ids=train_df['src'], dst_node_ids=train_df['dst'])
val_neg_sampler = NegativeEdgeSampler(src_node_ids=df['src'], dst_node_ids=df['dst'], seed=0)
nval_neg_sampler = NegativeEdgeSampler(src_node_ids=new_val_df['src'], dst_node_ids=new_val_df['dst'], seed=1)
test_neg_sampler = NegativeEdgeSampler(src_node_ids=df['src'], dst_node_ids=df['dst'], seed=2)
ntest_neg_sampler = NegativeEdgeSampler(src_node_ids=new_test_df['src'], dst_node_ids=new_test_df['dst'], seed=3)

def evaluate(eval_df, ngh_sampler, neg_sampler, mode='random'):
    neg_samples = 1
    model.eval()
    aps = list()
    aucs_mrrs = list()

    with torch.no_grad():
        total_loss = 0
        for _, rows in tqdm(eval_df.groupby(eval_df.index // train_param['batch_size'])):
            src, dst, ts = rows.src.values, rows.dst.values, rows.time.values

            # Follow DyGLib.
            bs = len(rows)
            num_neg = len(rows) * neg_samples
            if mode == 'random':
                _, neg_dst = neg_sampler.sample(num_neg)
                neg_src = src
            else:
                neg_src, neg_dst = neg_sampler.sample(num_neg, 
                    batch_src_node_ids=src,
                    batch_dst_node_ids=dst,
                    current_batch_start_time=ts[0],
                    current_batch_end_time=ts[-1])
            
            # Get embeddings of [src, dst, neg_src, neg_dst]
            root_nodes = np.concatenate([src, dst, neg_src, neg_dst]).astype(np.int64)
            ts = np.tile(ts, 4).astype(np.float32)
            if ngh_sampler is not None:
                if 'no_neg' in sample_param and sample_param['no_neg']:
                    pos_root_end = len(rows) * 2
                    ngh_sampler.sample(root_nodes[:pos_root_end], ts[:pos_root_end])
                else:
                    ngh_sampler.sample(root_nodes, ts)
                ret = ngh_sampler.get_ret()
            if gnn_param['arch'] != 'identity':
                mfgs = to_dgl_blocks(ret, sample_param['history'])
            else:
                mfgs = node_to_dgl_blocks(root_nodes, ts)
            mfgs = prepare_input(mfgs, nfeat, efeat, combine_first=combine_first)
            if mailbox is not None:
                mailbox.prep_input_mails(mfgs[0])
            embs = model.get_emb(mfgs)

            h_src, h_dst = embs[:bs], embs[bs:2*bs]
            h_nsrc, h_ndst = embs[2*bs:3*bs], embs[3*bs:4*bs]
            pred_pos, _ = model.edge_predictor(torch.cat([h_src, h_dst, h_dst]))
            pred_neg, _ = model.edge_predictor(torch.cat([h_nsrc, h_ndst, h_ndst]))

            # pred_pos, pred_neg = model(mfgs, neg_samples=neg_samples)
            total_loss += creterion(pred_pos, torch.ones_like(pred_pos))
            total_loss += creterion(pred_neg, torch.zeros_like(pred_neg))
            y_pred = torch.cat([pred_pos, pred_neg], dim=0).sigmoid().cpu()
            y_true = torch.cat([torch.ones(pred_pos.size(0)), torch.zeros(pred_neg.size(0))], dim=0)
            aps.append(average_precision_score(y_true, y_pred))
            if neg_samples > 1:
                aucs_mrrs.append(torch.reciprocal(torch.sum(pred_pos.squeeze() < pred_neg.squeeze().reshape(neg_samples, -1), dim=0) + 1).type(torch.float))
            else:
                aucs_mrrs.append(roc_auc_score(y_true, y_pred))
            if mailbox is not None:
                eid = rows['eid'].values
                mem_efeat = efeat[eid] if efeat is not None else None
                block = None
                if memory_param['deliver_to'] == 'neighbors':
                    block = to_dgl_blocks(ret, sample_param['history'], reverse=True)[0][0]
                # Set neg_samples=2 to fetch positive src and dst from root_nodes.
                mailbox.update_mailbox(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, root_nodes, ts, mem_efeat, block, neg_samples=2)
                mailbox.update_memory(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, root_nodes, model.memory_updater.last_updated_ts, neg_samples=2)
        val_losses.append(float(total_loss))
    ap = float(torch.tensor(aps).mean())
    if neg_samples > 1:
        auc_mrr = float(torch.cat(aucs_mrrs).mean())
    else:
        auc_mrr = float(torch.tensor(aucs_mrrs).mean())
    return ap, auc_mrr

if not os.path.isdir('saved_models'):
    os.makedirs('saved_models', exist_ok=True)
path_saver = f'saved_models/{args.data}_{args.train_neg}_{args.model_name}_{time.time()}.pth'
memory_saver = f'saved_models/{args.data}_{args.train_neg}_{args.model_name}_memory_{time.time()}.pth'
val_losses = list()

# The original implementation of random chunk scheduling is confusing.
# So we re-write this pipeline according to Algorithm 2 of TGL.
class RandomChunk(object):
    def __init__(self, seed=42):
        self.seed = seed
        self.random_state = np.random.RandomState(seed)
    
    def split_batch(self):
        # compute the original batch_size
        group_idx = np.arange(len(train_df)) // batch_size 
        # select a chunk in the first batch
        start_idx = int(self.random_state.rand() * (batch_size // chunk_size) * chunk_size)
        # split the first batch into two batches
        group_idx[start_idx:] += 1 
        return group_idx
    
random_chunk = RandomChunk(seed=42)
early_stopper = EarlyStopMonitor(max_round=50)

# Prepare the necessary variables for negative sampling.
if np.intersect1d(train_df.src.values, train_df.dst.values).size > 0.05 * train_df.src.nunique():
    node = np.concatenate((train_df.src.values, train_df.dst.values))
    min_dst, max_dst = int(min(node)), int(max(node))
    bipartite = False
else:
    bipartite = True
    train_df = train_df[~train_df.src.isin(np.intersect1d(train_df.src.values, train_df.dst.values))]
    min_dst, max_dst = int(min(train_df.dst.values)), int(max(train_df.dst.values))

train_neighbor_sampler = get_neighbor_sampler(data=train_df.astype(int), sample_neighbor_strategy=args.sample_neighbor_strategy, time_scaling_factor=args.time_scaling_factor, unique=args.unique)
epsilon = 1e-30
start_cache = args.num_epoch+1
num_hist = int(args.M*args.max_hist_prop)
args.max_anneal_epoch = min(args.max_anneal_epoch, train_param['epoch'])
args.thres_outer_policy, args.thres_inner_policy = ring_robust.create_negative_annealing_schedule(args)
min_src, max_src = int(min(train_df.src.values)), int(max(train_df.src.values))

Mu_nodes = np.zeros([max_src-min_src+1, args.M], dtype=int)
Mu_times = np.zeros([max_src-min_src+1, args.M], dtype=int)
score_all = []
logging.info('bipartite: %s.', bipartite)

def normalize(var):
    return (var - var.mean()) / var.std() + 1

for e in range(args.num_epoch):
    logging.info('Epoch {:d}:'.format(e))
    time_sample = 0
    time_prep = 0
    time_tot = 0
    total_loss = 0
    # training
    model.train()
    if train_ngh_sampler is not None:
        train_ngh_sampler.reset()
    if mailbox is not None:
        mailbox.reset()
        model.memory_updater.last_updated_nid = None

    time_cache = np.zeros(max_dst-min_dst+1, dtype=int)
    score_one_epoch = np.zeros((max_src-min_src+1,max_dst-min_dst+1))
    coo_mat = -1

    # Get pi size.
    thres_outer, thres_inner = args.thres_outer_policy.get_threshold(args.global_step), args.thres_inner_policy.get_threshold(args.global_step)
    ring_sampler = ring_robust.Ring(
                thres_outer=thres_outer,
                thres_inner=thres_inner,
                dynamic_inner=True
            )
    
    # Don't stop until the model attains some understanding of the relevant factors.
    if e <= args.stop_irr_epoch*6:
        early_stopper.clear()

    for _, rows in tqdm(train_df.groupby(random_chunk.split_batch())):
        t_tot_s = time.time()
        # Prepare input data
        batch_src, batch_dst, batch_time, batch_rows = rows.src.values.astype(int), rows.dst.values.astype(int), rows.time.values, len(rows)
        neg_samples = args.M
        num_cand = args.M*batch_rows
        num_hard, outer_idx, inner_idx = ring_sampler.compute_hard_num(num_cand)
        rand_prop = max(outer_idx/num_cand, args.min_rand_prop)

        # Determine cache start.
        if (start_cache == args.num_epoch+1) and (num_hard <= num_cand*args.cache_threshold):
            start_cache = e
            early_stopper.clear()

        # Generate the negative pool.
        if e <= start_cache:
            cand_nodes, neighbor_times = train_neighbor_sampler.sampler(node_ids=batch_src,
                                                                            dst_ids=batch_dst,
                                                                            node_interact_times=batch_time,
                                                                            num_neighbors=num_hist,
                                                                            num_random=args.M-num_hist)

        else:
            # Update the negative pool with half of the new samples and half from the cache.
            cand_nodes = Mu_nodes[batch_src-min_src]
            neighbor_times = Mu_times[batch_src-min_src]

        # Add random negative samples.
        neg_samples += args.M
        _, batch_rand_nodes = train_neg_sampler.random_sample(batch_rows*args.M)
        batch_rand_nodes = batch_rand_nodes.reshape(batch_rows, args.M)
        all_nodes = np.concatenate((cand_nodes, batch_rand_nodes), axis=1)
        all_nodes = all_nodes.flatten()
        cand_time = np.repeat(batch_time, 2*args.M)
        
        # Sample Message-passing Flow Graphs from root_nodes.
        root_nodes = np.concatenate([batch_src, batch_dst, all_nodes]).astype(np.int32)
        ts = np.concatenate([batch_time, batch_time, cand_time]).astype(np.float32)
        if train_ngh_sampler is not None:
            if 'no_neg' in sample_param and sample_param['no_neg']:
                pos_root_end = len(rows) * 2
                train_ngh_sampler.sample(root_nodes[:pos_root_end], ts[:pos_root_end])
            else:
                train_ngh_sampler.sample(root_nodes, ts)
            ret = train_ngh_sampler.get_ret()
            time_sample += ret[0].sample_time()
        t_prep_s = time.time()
        if gnn_param['arch'] != 'identity':
            mfgs = to_dgl_blocks(ret, sample_param['history'])
        else:
            mfgs = node_to_dgl_blocks(root_nodes, ts)
        mfgs = prepare_input(mfgs, nfeat, efeat, combine_first=combine_first)
        if mailbox is not None:
            mailbox.prep_input_mails(mfgs[0])
        time_prep += time.time() - t_prep_s
        # Model training.
        optimizer.zero_grad()

        emb = model.get_emb(mfgs)

        s_e = emb[:batch_rows]
        p_e = emb[batch_rows:2*batch_rows]

        # Obtain Time2Vec embeddings.
        s_temb = model.get_t2v1(torch.FloatTensor(rows.time.values).unsqueeze(1).to(args.device))
        n_interact_temb = model.get_t2v2(torch.tensor(neighbor_times.flatten()).unsqueeze(1).to(args.device).float())
        n_occurr_temb = model.get_t2v3(torch.FloatTensor(time_cache[cand_nodes.flatten()-min_dst]).unsqueeze(1).to(args.device))

        pred_pos, pred_neg_all = model.edge_predictor(torch.cat([s_e, p_e, emb[2*batch_rows:]]), neg_samples=2*args.M)
        pred_neg_all = pred_neg_all.view(batch_rows, 2*args.M)
        pred_neg = pred_neg_all[:, :args.M].reshape(batch_rows*args.M)
        pred_rand = pred_neg_all[:, args.M:].reshape(batch_rows*args.M)

        n_e = emb[2*batch_rows:(2+args.M)*batch_rows]
        n_e = n_e.view(batch_rows, args.M, dim_out)
        
        # Disentangle relevant and irrelevant factors.
        gate_p = torch.sigmoid(model.user_gate(s_e) + model.item_gate(p_e))

        gated_p_e_r = p_e * gate_p * normalize(s_temb) # [batch_size, channel]
        gated_p_e_ir = p_e - gated_p_e_r

        gate_n = torch.sigmoid(model.pos_gate(gated_p_e_r).unsqueeze(1) + model.neg_gate(n_e))

        gated_n_e_r = n_e * gate_n * normalize((n_interact_temb + n_occurr_temb).view(batch_rows, args.M, dim_out))   # [batch_size, n_negs, channel]
        gated_n_e_ir = n_e - gated_n_e_r

        gated_n_e_r = gated_n_e_r.view(batch_rows * args.M, dim_out)
        gated_n_e_ir = gated_n_e_ir.view(batch_rows * args.M, dim_out)
        
        # Selection function.
        gated_pos_scores_r, gated_neg_scores_r = model.edge_predictor(torch.cat([s_e, gated_p_e_r, gated_n_e_r]), neg_samples=args.M)
        gated_pos_scores_ir, gated_neg_scores_ir = model.edge_predictor(torch.cat([s_e, gated_p_e_ir, gated_n_e_ir]), neg_samples=args.M)
        gated_neg_scores_r = gated_neg_scores_r.view(batch_rows, args.M)
        gated_neg_scores_ir = gated_neg_scores_ir.view(batch_rows, args.M)

        scoring_sel = - min(1, e / args.stop_irr_epoch) * torch.abs(gated_pos_scores_r - gated_neg_scores_r) - (2 - min(1, e / args.stop_irr_epoch)) * torch.abs(gated_neg_scores_ir - gated_pos_scores_ir)    # [batch_size, n_negs, channel]
        
        # Sample based on Adaptive Pi.
        pred_sel = pred_neg.flatten()[torch.argsort(scoring_sel.flatten(), descending=True)[inner_idx:outer_idx]].view(num_hard,1)

        # Train using selected and random negative samples.
        loss = creterion(pred_pos, torch.ones_like(pred_pos))
        loss += (1-rand_prop) * creterion(pred_sel, torch.zeros_like(pred_sel))
        loss += rand_prop * creterion(pred_rand, torch.zeros_like(pred_rand))

        # Contrastive loss.
        loss -= args.lbd * (torch.mean(torch.log(epsilon + torch.sigmoid(gated_pos_scores_r - gated_pos_scores_ir))) +
                            torch.mean(torch.log(epsilon + torch.sigmoid(gated_neg_scores_ir - gated_neg_scores_r))) +
                            torch.mean(torch.log(epsilon + torch.sigmoid(gated_pos_scores_r - gated_neg_scores_r))) +
                            torch.mean(torch.log(epsilon + torch.sigmoid(gated_neg_scores_ir - gated_pos_scores_ir)))) / 4
    
        # Regularize the Time2Vec embeddings.
        regularize = (torch.norm(s_temb) ** 2
                    + torch.norm(n_interact_temb) ** 2
                    + torch.norm(n_occurr_temb) ** 2) / 2
        
        loss += args.mu * regularize / batch_size

        total_loss += float(loss) * batch_size
        loss.backward()
        optimizer.step()
        t_prep_s = time.time()

        # Update mailbox.
        if mailbox is not None:
            eid = rows['eid'].values
            mem_efeat = efeat[eid] if efeat is not None else None
            block = None
            if memory_param['deliver_to'] == 'neighbors':
                block = to_dgl_blocks(ret, sample_param['history'], reverse=True)[0][0]
            if mailbox is not None:
                mailbox.update_mailbox(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, root_nodes, ts, mem_efeat, block, neg_samples=neg_samples)
                mailbox.update_memory(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, root_nodes, model.memory_updater.last_updated_ts, neg_samples=neg_samples)

        # Update the latest interaction timestamps for all nodes.
        valid_src = (batch_src <= max_dst) & (batch_src >= min_dst)
        np.maximum.at(time_cache, batch_src[valid_src]-min_dst, batch_time[valid_src])
        np.maximum.at(time_cache, batch_dst-min_dst, batch_time)

        # Update cache.
        if e >= start_cache:
            pred_neg = pred_neg.view(batch_rows, args.M).detach().cpu().numpy()
            score_row_idx = np.repeat(batch_src-min_src, args.M)
            score_col_idx = cand_nodes.flatten()-min_dst
            score_one_epoch[score_row_idx, score_col_idx] = pred_neg.flatten()
            if e-start_cache<5:
                update_scores = pred_neg
            else:
                score_batch = np.zeros((args.M*batch_rows, 5))
                for i in range(len(score_all)):
                    score_batch[:, i] = score_all[i].tocsr()[score_row_idx, score_col_idx].A.flatten()
                score_batch[score_batch == 0] = np.nan
                std_batch = np.nan_to_num(np.nanstd(score_batch, axis=1)).reshape(batch_rows, args.M)

                update_scores = pred_neg - args.std_weight * min(1,e/args.num_epoch) * std_batch
                del score_batch
            if args.fixed_update:
                sample_indices = np.argsort(-update_scores)[:, :args.M//2].flatten()
            else:
                probs = softmax(update_scores, axis=1)
                sample_indices = np.array([np.random.choice(probs.shape[1], size=args.M//2, p=probs[i], replace=False) for i in range(batch_rows)]).flatten() #without replacement

            Mu_row_idx = np.repeat(np.arange(batch_rows), args.M//2)
            Mu_nodes[batch_src-min_src, :args.M//2] = cand_nodes[Mu_row_idx, sample_indices].reshape(batch_rows, args.M//2)
            Mu_times[batch_src-min_src, :args.M//2] = neighbor_times[Mu_row_idx, sample_indices].reshape(batch_rows, args.M//2)

            new_nodes, new_times = train_neighbor_sampler.sampler(node_ids=batch_src,
                                                            dst_ids=batch_dst,
                                                            node_interact_times=batch_time,
                                                            num_neighbors=max(num_hist//2, 1),
                                                            num_random=args.M//2-max(num_hist//2, 1))
            Mu_nodes[batch_src-min_src, args.M//2:] = new_nodes
            Mu_times[batch_src-min_src, args.M//2:] = new_times

            del _, update_scores, sample_indices, score_row_idx, score_col_idx, Mu_row_idx, new_nodes, new_times

        time_prep += time.time() - t_prep_s
        time_tot += time.time() - t_tot_s

    if e-start_cache>=5:
        del std_batch
    coo_mat = coo_matrix(score_one_epoch)
    del score_one_epoch

    score_all = score_all[-4:]
    score_all.append(coo_mat)
    del coo_mat

    # Advance full_ngh_sampler.update_ts_ptr().
    if full_ngh_sampler is not None:
        full_ngh_sampler.reset()
    if mailbox is not None:
        model.memory_updater.last_updated_nid = None
        mailbox.eval()
        mailbox.reset()
    
    ap, auc = evaluate(train_df, full_ngh_sampler, train_neg_sampler)
    logging.info('\ttrain_ap:{:4f}  train_auc:{:4f}'.format(ap, auc))

    # Update pi based on average precision.
    if args.anneal_mode == "adaptive_anneal_on_loss":
        args.thres_outer_policy.record(ap)
    args.global_step += 1

    ap, auc = evaluate(val_df, full_ngh_sampler, val_neg_sampler)

    # Save the best model using early stopping.
    if early_stopper.early_stop_check(ap):
        break
    if early_stopper.best_epoch == e:
        torch.save(model.state_dict(), path_saver)
    logging.info('\ttrain loss:{:.4f}  val_ap:{:4f}  val_auc:{:4f}'.format(total_loss / len(train_df), ap, auc))
    logging.info('\ttotal time:{:.2f}s sample time:{:.2f}s prep time:{:.2f}s'.format(time_tot, time_sample, time_prep))

logging.info('Loading model at epoch %d.', early_stopper.best_epoch)
model.load_state_dict(torch.load(path_saver))
model.eval()
if full_ngh_sampler is not None:
    full_ngh_sampler.reset()
if mailbox is not None:
    model.memory_updater.last_updated_nid = None
    mailbox.eval()
    mailbox.reset()
evaluate(train_df, full_ngh_sampler, train_neg_sampler)

# Three negative testing strategies including random, historical, inductive for transductive learning and inductive learning.
logging.info('********** Random negative testing. **********')
val_neg_sampler = NegativeEdgeSampler(src_node_ids=df['src'], dst_node_ids=df['dst'], seed=0)
nval_neg_sampler = NegativeEdgeSampler(src_node_ids=new_val_df['src'], dst_node_ids=new_val_df['dst'], seed=1)
test_neg_sampler = NegativeEdgeSampler(src_node_ids=df['src'], dst_node_ids=df['dst'], seed=2)
ntest_neg_sampler = NegativeEdgeSampler(src_node_ids=new_test_df['src'], dst_node_ids=new_test_df['dst'], seed=3)

val_ap, val_auc = evaluate(val_df, full_ngh_sampler, val_neg_sampler)
nval_ap, nval_auc = evaluate(new_val_df, full_ngh_sampler, nval_neg_sampler)
# Save the memory state in the validation stage.
if mailbox is not None:
    memory_state = mailbox.state_dict()
    updater_state = model.backup_memory_updater()
test_ap, test_auc = evaluate(test_df, full_ngh_sampler, test_neg_sampler)
# Recover the memory state till the validation stage.
if mailbox is not None:
    mailbox.load_state_dict(memory_state)
    model.load_memory_updater(updater_state)
ntest_ap, ntest_auc = evaluate(new_test_df, full_ngh_sampler, ntest_neg_sampler)

logging.info('val_ap: %.4f, val_auc: %.4f, nval_ap: %.4f, nval_auc: %.4f', val_ap, val_auc, nval_ap, nval_auc)
logging.info('test_ap: %.4f, test_auc: %.4f, ntest_ap: %.4f, ntest_auc: %.4f', test_ap, test_auc, ntest_ap, ntest_auc)

# Reset ts_ptr of full_ngh_sampler.
if full_ngh_sampler is not None:
    full_ngh_sampler.reset()
if mailbox is not None:
    model.memory_updater.last_updated_nid = None
    mailbox.eval()
    mailbox.reset()
evaluate(train_df, full_ngh_sampler, train_neg_sampler)

# Follow DGB and DyGLib.
logging.info('********** Historical negative testing. **********')
val_params = {'src_node_ids': df['src'], 'dst_node_ids': df['dst'], 'interact_times': df['time'], 'last_observed_time': train_df['time'].iloc[-1], 'negative_sample_strategy':'historical', 'seed': 0}
nval_params = {'src_node_ids': new_val_df['src'], 'dst_node_ids': new_val_df['dst'], 'interact_times': new_val_df['time'], 'last_observed_time': train_df['time'].iloc[-1], 'negative_sample_strategy':'historical', 'seed': 1}
test_params = {'src_node_ids': df['src'], 'dst_node_ids': df['dst'], 'interact_times': df['time'], 'last_observed_time': val_df['time'].iloc[-1], 'negative_sample_strategy':'historical', 'seed': 2}
ntest_params = {'src_node_ids': new_test_df['src'], 'dst_node_ids': new_test_df['dst'], 'interact_times': new_test_df['time'], 'last_observed_time': val_df['time'].iloc[-1], 'negative_sample_strategy':'historical', 'seed': 3}

val_neg_sampler = NegativeEdgeSampler(**val_params)
nval_neg_sampler = NegativeEdgeSampler(**nval_params)
test_neg_sampler = NegativeEdgeSampler(**test_params)
ntest_neg_sampler = NegativeEdgeSampler(**ntest_params)

hist_val_ap, hist_val_auc = evaluate(val_df, full_ngh_sampler, val_neg_sampler, mode='historical')
hist_nval_ap, hist_nval_auc = evaluate(new_val_df, full_ngh_sampler, nval_neg_sampler, mode='historical')
# Save the memory state in the validation stage.
if mailbox is not None:
    memory_state = mailbox.state_dict()
    updater_state = model.backup_memory_updater()
hist_test_ap, hist_test_auc = evaluate(test_df, full_ngh_sampler, test_neg_sampler, mode='historical')
# Recover the memory state till the validation stage.
if mailbox is not None:
    mailbox.load_state_dict(memory_state)
    model.load_memory_updater(updater_state)
hist_ntest_ap, hist_ntest_auc = evaluate(new_test_df, full_ngh_sampler, ntest_neg_sampler, mode='historical')

logging.info('hist_val_ap: %.4f, hist_val_auc: %.4f, hist_nval_ap: %.4f, hist_nval_auc: %.4f', hist_val_ap, hist_val_auc, hist_nval_ap, hist_nval_auc)
logging.info('hist_test_ap: %.4f, hist_test_auc: %.4f, hist_ntest_ap: %.4f, hist_ntest_auc: %.4f', hist_test_ap, hist_test_auc, hist_ntest_ap, hist_ntest_auc)

# Reset ts_ptr of full_ngh_sampler.
if full_ngh_sampler is not None:
    full_ngh_sampler.reset()
if mailbox is not None:
    model.memory_updater.last_updated_nid = None
    mailbox.eval()
    mailbox.reset()
evaluate(train_df, full_ngh_sampler, train_neg_sampler)

logging.info('********** Half Random Half Historical negative testing. **********')
val_params = {'src_node_ids': df['src'], 'dst_node_ids': df['dst'], 'interact_times': df['time'], 'last_observed_time': train_df['time'].iloc[-1], 'negative_sample_strategy':'half_historical', 'seed': 0}
nval_params = {'src_node_ids': new_val_df['src'], 'dst_node_ids': new_val_df['dst'], 'interact_times': new_val_df['time'], 'last_observed_time': train_df['time'].iloc[-1], 'negative_sample_strategy':'half_historical', 'seed': 1}
test_params = {'src_node_ids': df['src'], 'dst_node_ids': df['dst'], 'interact_times': df['time'], 'last_observed_time': val_df['time'].iloc[-1], 'negative_sample_strategy':'half_historical', 'seed': 2}
ntest_params = {'src_node_ids': new_test_df['src'], 'dst_node_ids': new_test_df['dst'], 'interact_times': new_test_df['time'], 'last_observed_time': val_df['time'].iloc[-1], 'negative_sample_strategy':'half_historical', 'seed': 3}

val_neg_sampler = NegativeEdgeSampler(**val_params)
nval_neg_sampler = NegativeEdgeSampler(**nval_params)
test_neg_sampler = NegativeEdgeSampler(**test_params)
ntest_neg_sampler = NegativeEdgeSampler(**ntest_params)

half_hist_val_ap, half_hist_val_auc = evaluate(val_df, full_ngh_sampler, val_neg_sampler, mode='half_historical')
half_hist_nval_ap, half_hist_nval_auc = evaluate(new_val_df, full_ngh_sampler, nval_neg_sampler, mode='half_historical')
# Save the memory state in the validation stage.
if mailbox is not None:
    memory_state = mailbox.state_dict()
    updater_state = model.backup_memory_updater()
half_hist_test_ap, half_hist_test_auc = evaluate(test_df, full_ngh_sampler, test_neg_sampler, mode='half_historical')
# Recover the memory state till the validation stage.
if mailbox is not None:
    mailbox.load_state_dict(memory_state)
    model.load_memory_updater(updater_state)
half_hist_ntest_ap, half_hist_ntest_auc = evaluate(new_test_df, full_ngh_sampler, ntest_neg_sampler, mode='half_historical')

logging.info('half_hist_val_ap: %.4f, half_hist_val_auc: %.4f, half_hist_nval_ap: %.4f, half_hist_nval_auc: %.4f', half_hist_val_ap, half_hist_val_auc, half_hist_nval_ap, half_hist_nval_auc)
logging.info('half_hist_test_ap: %.4f, half_hist_test_auc: %.4f, half_hist_ntest_ap: %.4f, half_hist_ntest_auc: %.4f', half_hist_test_ap, half_hist_test_auc, half_hist_ntest_ap, half_hist_ntest_auc)

# Reset ts_ptr of full_ngh_sampler.
if full_ngh_sampler is not None:
    full_ngh_sampler.reset()
if mailbox is not None:
    model.memory_updater.last_updated_nid = None
    mailbox.eval()
    mailbox.reset()
evaluate(train_df, full_ngh_sampler, train_neg_sampler)

logging.info('********** Inductive negative testing. **********')
for params in [val_params, nval_params, test_params, ntest_params]:
    params['negative_sample_strategy'] = 'inductive'

val_neg_sampler = NegativeEdgeSampler(**val_params)
nval_neg_sampler = NegativeEdgeSampler(**nval_params)
test_neg_sampler = NegativeEdgeSampler(**test_params)
ntest_neg_sampler = NegativeEdgeSampler(**ntest_params)

indu_val_ap, indu_val_auc = evaluate(val_df, full_ngh_sampler, val_neg_sampler, mode='inductive')
indu_nval_ap, indu_nval_auc = evaluate(new_val_df, full_ngh_sampler, nval_neg_sampler, mode='inductive')
# Save the memory state in the validation stage.
if mailbox is not None:
    memory_state = mailbox.state_dict()
    updater_state = model.backup_memory_updater()
indu_test_ap, indu_test_auc = evaluate(test_df, full_ngh_sampler, test_neg_sampler, mode='inductive')
# Recover the memory state till the validation stage.
if mailbox is not None:
    mailbox.load_state_dict(memory_state)
    model.load_memory_updater(updater_state)
indu_ntest_ap, indu_ntest_auc = evaluate(new_test_df, full_ngh_sampler, ntest_neg_sampler, mode='inductive')

logging.info('indu_val_ap: %.4f, indu_val_auc: %.4f, indu_nval_ap: %.4f, indu_nval_auc: %.4f', indu_val_ap, indu_val_auc, indu_nval_ap, indu_nval_auc)
logging.info('indu_test_ap: %.4f, indu_test_auc: %.4f, indu_ntest_ap: %.4f, indu_ntest_auc: %.4f', indu_test_ap, indu_test_auc, indu_ntest_ap, indu_ntest_auc)

SAVE_DIR = 'saved_results/'
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR, exist_ok=True)
headers = ['dataset', 'method', 'train_neg',
           'rand_val_ap', 'rand_val_auc', 'rand_nval_ap', 'rand_nval_auc', 
           'hist_val_ap', 'hist_val_auc', 'hist_nval_ap', 'hist_nval_auc',
           'half_hist_val_ap', 'half_hist_val_auc', 'half_hist_nval_ap', 'half_hist_nval_auc',
           'indu_val_ap', 'indu_val_auc', 'indu_nval_ap', 'indu_nval_auc',
           'rand_test_ap', 'rand_test_auc', 'rand_ntest_ap', 'rand_ntest_auc', 
           'hist_test_ap', 'hist_test_auc', 'hist_ntest_ap', 'hist_ntest_auc',
           'half_hist_test_ap', 'half_hist_test_auc', 'half_hist_ntest_ap', 'half_hist_ntest_auc',
           'indu_test_ap', 'indu_test_auc', 'indu_ntest_ap', 'indu_ntest_auc',
            'params']
result_path = f'{SAVE_DIR}/{args.data}-{args.model_name}.csv'
if not os.path.exists(result_path):
    with open(result_path, 'w+') as fw:
        fw.write(','.join(headers) + '\r\n')
        os.chmod(result_path, 0o777)
config = f'seed={args.seed}'
with open(result_path, 'a') as fw:
    fw.write(f'{args.data},{args.model_name},{args.train_neg},' +
             ','.join([f'{el:.4f}' for el in [val_ap, val_auc, nval_ap, nval_auc]]) + ',' +
             ','.join([f'{el:.4f}' for el in [hist_val_ap, hist_val_auc, hist_nval_ap, hist_nval_auc]]) + ',' +
             ','.join([f'{el:.4f}' for el in [half_hist_val_ap, half_hist_val_auc, half_hist_nval_ap, half_hist_nval_auc]]) + ',' +
             ','.join([f'{el:.4f}' for el in [indu_val_ap, indu_val_auc, indu_nval_ap, indu_nval_auc]]) + ',' +
             ','.join([f'{el:.4f}' for el in [test_ap, test_auc, ntest_ap, ntest_auc]]) + ',' +
             ','.join([f'{el:.4f}' for el in [hist_test_ap, hist_test_auc, hist_ntest_ap, hist_ntest_auc]]) + ',' +
             ','.join([f'{el:.4f}' for el in [half_hist_test_ap, half_hist_test_auc, half_hist_ntest_ap, half_hist_ntest_auc]]) + ',' +
             ','.join([f'{el:.4f}' for el in [indu_test_ap, indu_test_auc, indu_ntest_ap, indu_ntest_auc]]) + ',' +
             f'"{config}"')
    fw.write('\r\n')