from datetime import datetime
import logging
import random

import torch
import os
import yaml
import dgl
import time
import pandas as pd
from pytz import timezone, utc
import numpy as np
import numba


def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def write_result(val_metrics,
                 metrics,
                 dataset,
                 params,
                 postfix='TNS',
                 results='saved_results'):
    res_path = "{}/{}-{}.csv".format(results, dataset, postfix)
    val_keys = val_metrics.keys()
    test_keys = metrics.keys()
    headers = ["method", "dataset"
               ] + list(val_keys) + list(test_keys) + ["params"]
    if not os.path.exists(res_path):
        f = open(res_path, 'w')
        f.write(",".join(headers) + "\r\n")
        f.close()
        os.chmod(res_path, 0o777)
    with open(res_path, 'a') as f:
        result_str = "{},{}".format(postfix, dataset)
        result_str += "," + ",".join(
            ["{:.4f}".format(val_metrics[k]) for k in val_keys])
        result_str += "," + ",".join(
            ["{:.4f}".format(metrics[k]) for k in test_keys])
        logging.info(result_str)
        params_str = ",".join(
            ["{}={}".format(k, v) for k, v in params.items()])
        params_str = "\"{}\"".format(params_str)
        row = result_str + "," + params_str + "\r\n"
        f.write(row)

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print("%r  %2.2f s" % (method.__name__, te - ts))
        return result

    return timed


def set_logger(method, config_str, log_file=False):
    if log_file:
        os.makedirs('logs', exist_ok=True)
    numba_logger = logging.getLogger('numba')
    numba_logger.setLevel(logging.WARNING)
    
    def customTime(*args):
        utc_dt = utc.localize(datetime.utcnow())
        my_tz = timezone("Asia/Shanghai")
        converted = utc_dt.astimezone(my_tz)
        return converted.timetuple()

    logging.Formatter.converter = customTime

    # set up logger
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        "%Y-%m-%d %H:%M:%S")
    if log_file:
        fh = logging.FileHandler('logs/{}-{}-{}.log'.format(
            method,
            config_str,
            datetime.now(timezone('Asia/Shanghai')).strftime("%Y-%m-%d %H:%M:%S")))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger


class EarlyStopMonitor(object):
    def __init__(self, max_round=50, higher_better=True, tolerance=1e-3):
        self.max_round = max_round
        self.num_round = 0

        self.epoch_count = 0
        self.best_epoch = 0

        self.last_best = None
        self.higher_better = higher_better
        self.tolerance = tolerance
    
    def clear(self):
        self.num_round = 0

    def early_stop_check(self, curr_val):

        if not self.higher_better:
            curr_val *= -1
        if self.last_best is None:
            self.last_best = curr_val
        elif (curr_val - self.last_best) / np.abs(
                self.last_best) > self.tolerance:
            self.last_best = curr_val
            self.num_round = 0
            self.best_epoch = self.epoch_count
        else:
            self.num_round += 1
        self.epoch_count += 1
        return self.num_round >= self.max_round

@numba.jit(nopython=True, nogil=True)
def random_sample_with_collision_check(unq_src: np.ndarray, unq_dst: np.ndarray, size: int, batch_edges: np.ndarray, random_edge_indices: np.ndarray):
    '''We mitigate the issue of memory OOM by trading time for space. We use the iterative check
    over the product of (unique_src_node_ids, unique_dst_node_ids) to eliminate the batch_edges
    and randomly sample the left possible edges. 
    '''
    possible_edge_size = len(unq_src) * len(unq_dst) - len(batch_edges)
    assert possible_edge_size > 0

    # For each unique_src_node_id, we calculate its begin (inclusive) and end (exclusive) indices.
    possible_nghs = np.zeros((len(unq_src),), dtype=np.int64)
    for i in range(len(unq_src)):
        mask = batch_edges[:, 0] == unq_src[i]
        batch_ngh = batch_edges[mask][:, 1]
        possible_nghs[i] = len(unq_dst) - len(batch_ngh)
    
    # For each random_edge_index, we firstly find the corresponding src node, and then compute the corresponding dst node.
    cum_possible_indices = np.cumsum(possible_nghs)
    assert cum_possible_indices[-1] == possible_edge_size
    sample_src_idxs = np.zeros((size,), dtype=np.int64)
    sample_srcs = np.zeros((size,), dtype=np.int64)
    for i, eidx in enumerate(random_edge_indices):
        src_idx = np.searchsorted(cum_possible_indices, eidx, side='right')
        sample_src_idxs[i] = src_idx
        sample_srcs[i] = unq_src[src_idx]
    
    unq_dst = unq_dst.astype(np.int64)
    batch_edges = batch_edges.astype(np.int64)
    sample_dsts = np.zeros((size,), dtype=np.int64)
    for i in range(len(sample_src_idxs)):
        src, src_idx, eidx = sample_srcs[i], sample_src_idxs[i], random_edge_indices[i]
        mask = batch_edges[:, 0] == src
        batch_ngh = batch_edges[mask][:, 1]
        left_nghs = setdiff1d_nb(unq_dst, batch_ngh)
        if src_idx > 0:
            # Compute the relative neighbor index in the current src node. 
            cur_idx = eidx - cum_possible_indices[src_idx - 1]
        else:
            cur_idx = eidx
        dst_idx = left_nghs[cur_idx]
        sample_dsts[i] = dst_idx
    return sample_srcs, sample_dsts

@numba.jit(nopython=True, nogil=True)
def encode_edges(src_ids: np.ndarray[np.int64], dst_ids: np.ndarray[np.int64], max_nid: np.int64) -> np.ndarray[np.int64]:
    base = int(np.ceil(np.log(max_nid + 1) / np.log(10)))
    base10 = 10 ** base
    eidx = src_ids * base10 + dst_ids
    return eidx, base10

@numba.jit(nopython=True, nogil=True)
def decode_edges(eidx: np.ndarray[np.int64], base10: np.int64) -> np.ndarray[np.int64]:
    return np.stack((eidx // base10, eidx % base10)).T

@numba.jit(nopython=True, nogil=True)
def get_unique_edges(src_ids: np.ndarray[np.int64], dst_ids: np.ndarray[np.int64], max_nid: int) -> np.ndarray[np.int64]:
    selected_eidx, base10 = encode_edges(src_ids, dst_ids, max_nid)
    unique_eidx = np.unique(selected_eidx)
    unique_edges = decode_edges(unique_eidx, base10)

    return unique_edges

@numba.jit(nopython=True, nogil=True)
def get_unique_edges_between_start_end_time(start_time: np.float64, end_time: np.float64, interact_times: np.ndarray[np.float64], src_node_ids: np.ndarray[np.int64], dst_node_ids: np.ndarray[np.int64], max_nid: np.int64):
        """
        get unique edges happened between start and end time
        :param start_time: float, start timestamp
        :param end_time: float, end timestamp
        :return: a set of edges, where each edge is a tuple of (src_node_id, dst_node_id)
        """
        selected_time_interval = np.logical_and(interact_times >= start_time, interact_times <= end_time)
        # return the unique select source and destination nodes in the selected time interval
        # Speed up with numpy.
        selected_edges = np.stack((src_node_ids[selected_time_interval], dst_node_ids[selected_time_interval])).T
        assert selected_edges.shape[1] == 2
        return get_unique_edges(selected_edges[:, 0], selected_edges[:, 1], max_nid)

@numba.njit('int64[:](int64[:], int64[:])')
def setdiff1d_nb(arr1, arr2):
    delta = set(arr2)

    # : build the result
    result = np.empty(len(arr1), dtype=arr1.dtype)
    j = 0
    for i in range(arr1.shape[0]):
        if arr1[i] not in delta:
            result[j] = arr1[i]
            j += 1
    return result[:j]

@numba.jit(nopython=True, nogil=True)
def get_diff_edges(source_edges: np.ndarray, target_edges: np.ndarray, max_nid: np.int64):
        # For each pair of an edge, we map the edge into edge idx by src_node * 10_BASE + dst_node,
        # where 10_BASE = int(np.ceil(np.log(dst_nodes.max() + 1) / np.log(10))).
        source_edges = source_edges.astype(np.int64)
        target_edges = target_edges.astype(np.int64)
        assert source_edges.shape[1] == 2 and target_edges.shape[1] == 2
        src_eidx, base10 = encode_edges(source_edges[:, 0], source_edges[:, 1], max_nid)
        tgt_idx, _ = encode_edges(target_edges[:, 0], target_edges[:, 1], max_nid)
        # We assume source edges and target edges are both unique.
        diff_idx = setdiff1d_nb(src_eidx, tgt_idx)
        diff_edges = decode_edges(diff_idx, base10)
        assert diff_edges.shape[1] == 2
        return diff_edges


class NegativeEdgeSampler(object):
    """
    Copyright (c) 2023 DyGLib https://github.com/yule-BUAA/DyGLib/blob/master/utils/utils.py#L305

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    """

    def __init__(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, interact_times: np.ndarray = None, last_observed_time: float = None,
                 negative_sample_strategy: str = 'random', seed: int = None):
        """
        Negative Edge Sampler, which supports three strategies: "random", "historical", "inductive".
        :param src_node_ids: ndarray, (num_src_nodes, ), source node ids, num_src_nodes == num_dst_nodes
        :param dst_node_ids: ndarray, (num_dst_nodes, ), destination node ids
        :param interact_times: ndarray, (num_src_nodes, ), interaction timestamps
        :param last_observed_time: float, time of the last observation (for inductive negative sampling strategy)
        :param negative_sample_strategy: str, negative sampling strategy, can be "random", "historical", "inductive"
        :param seed: int, random seed
        """
        self.seed = seed
        self.negative_sample_strategy = negative_sample_strategy
        self.src_node_ids = src_node_ids.values.astype(np.int64)
        self.dst_node_ids = dst_node_ids.values.astype(np.int64)
        if interact_times is not None:
            self.interact_times = interact_times.values.astype(np.float64)
        else:
            self.interact_times = interact_times
        self.unique_src_node_ids = np.unique(src_node_ids).astype(np.int64)
        self.unique_dst_node_ids = np.unique(dst_node_ids).astype(np.int64)
        self.max_nid = int(max(src_node_ids.max(), dst_node_ids.max()))
        self.unique_interact_times = np.unique(interact_times)
        self.earliest_time = min(self.unique_interact_times)
        self.last_observed_time = last_observed_time

        # We use random_sample_with_collision_check to avoid OOM
        if self.negative_sample_strategy != 'random':
            # all the possible edges that connect source nodes in self.unique_src_node_ids with destination nodes in self.unique_dst_node_ids
            self.possible_edges = set((src_node_id, dst_node_id) for src_node_id in self.unique_src_node_ids for dst_node_id in self.unique_dst_node_ids)

        if self.negative_sample_strategy == 'inductive':
            # set of observed edges
            self.observed_edges = self.get_unique_edges_between_start_end_time(self.earliest_time, self.last_observed_time)

        if self.seed is not None:
            self.random_state = np.random.RandomState(self.seed)

    def get_unique_edges_between_start_end_time(self, start_time: float, end_time: float, mode='old'):
        """
        get unique edges happened between start and end time
        :param start_time: float, start timestamp
        :param end_time: float, end timestamp
        :return: a set of edges, where each edge is a tuple of (src_node_id, dst_node_id)
        """
        if mode == 'new':
            return get_unique_edges_between_start_end_time(start_time, end_time, self.interact_times, self.src_node_ids, self.dst_node_ids, self.max_nid)

        selected_time_interval = np.logical_and(self.interact_times >= start_time, self.interact_times <= end_time)
        # return the unique select source and destination nodes in the selected time interval
        return set((src_node_id, dst_node_id) for src_node_id, dst_node_id in zip(self.src_node_ids[selected_time_interval], self.dst_node_ids[selected_time_interval]))
    
    
    def sample(self, size: int, batch_src_node_ids: np.ndarray = None, batch_dst_node_ids: np.ndarray = None,
               current_batch_start_time: float = 0.0, current_batch_end_time: float = 0.0):
        """
        sample negative edges, support random, historical and inductive sampling strategy
        :param size: int, number of sampled negative edges
        :param batch_src_node_ids: ndarray, shape (batch_size, ), source node ids in the current batch
        :param batch_dst_node_ids: ndarray, shape (batch_size, ), destination node ids in the current batch
        :param current_batch_start_time: float, start time in the current batch
        :param current_batch_end_time: float, end time in the current batch
        :return:
        """
        if self.negative_sample_strategy == 'random':
            negative_src_node_ids, negative_dst_node_ids = self.random_sample(size=size)
        elif self.negative_sample_strategy == 'historical':
            negative_src_node_ids, negative_dst_node_ids = self.historical_sample(size=size, batch_src_node_ids=batch_src_node_ids,
             batch_dst_node_ids=batch_dst_node_ids,
             current_batch_start_time=current_batch_start_time,
             current_batch_end_time=current_batch_end_time)
        elif self.negative_sample_strategy == 'half_historical':
            rand_negative_src_node_ids, rand_negative_dst_node_ids = self.random_sample(size=size//2)
            hist_negative_src_node_ids, hist_negative_dst_node_ids = self.historical_sample(size=size-size//2, batch_src_node_ids=batch_src_node_ids,
             batch_dst_node_ids=batch_dst_node_ids,
             current_batch_start_time=current_batch_start_time,
             current_batch_end_time=current_batch_end_time)
            negative_src_node_ids = np.concatenate([rand_negative_src_node_ids, hist_negative_src_node_ids])
            negative_dst_node_ids = np.concatenate([rand_negative_dst_node_ids, hist_negative_dst_node_ids])
        elif self.negative_sample_strategy == 'inductive':
            negative_src_node_ids, negative_dst_node_ids = self.inductive_sample(size=size, batch_src_node_ids=batch_src_node_ids,
             batch_dst_node_ids=batch_dst_node_ids,
             current_batch_start_time=current_batch_start_time,
             current_batch_end_time=current_batch_end_time)
        else:
            raise ValueError(f'Not implemented error for negative_sample_strategy {self.negative_sample_strategy}!')
        return negative_src_node_ids, negative_dst_node_ids

    def random_sample(self, size: int):
        """
        random sampling strategy, which is used by previous works
        :param size: int, number of sampled negative edges
        :return:
        """
        if self.seed is None:
            random_sample_edge_src_node_indices = np.random.randint(0, len(self.unique_src_node_ids), size)
            random_sample_edge_dst_node_indices = np.random.randint(0, len(self.unique_dst_node_ids), size)
        else:
            random_sample_edge_src_node_indices = self.random_state.randint(0, len(self.unique_src_node_ids), size)
            random_sample_edge_dst_node_indices = self.random_state.randint(0, len(self.unique_dst_node_ids), size)
        return self.unique_src_node_ids[random_sample_edge_src_node_indices], self.unique_dst_node_ids[random_sample_edge_dst_node_indices]

    def random_sample_with_collision_check(self, size: int, batch_src_node_ids: np.ndarray, batch_dst_node_ids: np.ndarray, mode='old'):
        """
        random sampling strategy with collision check, which guarantees that the sampled edges do not appear in the current batch,
        used for historical and inductive sampling strategy
        :param size: int, number of sampled negative edges
        :param batch_src_node_ids: ndarray, shape (batch_size, ), source node ids in the current batch
        :param batch_dst_node_ids: ndarray, shape (batch_size, ), destination node ids in the current batch
        :return:
        """
        # The original self.possible_edges stores N*N edges by iterating over the product of nodes,
        # which causes OOM on a graph with over 100K nodes. So we use the additional computation to
        # avoid the huge memory footprint to sample from possible_random_edges.
        batch_edges = np.unique(np.stack((batch_src_node_ids, batch_dst_node_ids)).T, axis=0).astype(np.int64)
        possible_edge_size = len(self.unique_src_node_ids) * len(self.unique_dst_node_ids) - len(batch_edges)
        assert possible_edge_size > 0

        assert batch_src_node_ids is not None and batch_dst_node_ids is not None

        if mode == 'new':
            # if replace is True, then a value in the list can be selected multiple times, otherwise, a value can be selected only once at most
            random_edge_indices = self.random_state.choice(possible_edge_size, size=size, replace=(possible_edge_size < size))
            return random_sample_with_collision_check(self.unique_src_node_ids, self.unique_dst_node_ids, size, batch_edges, random_edge_indices)

        batch_edges = set((batch_src_node_id, batch_dst_node_id) for batch_src_node_id, batch_dst_node_id in zip(batch_src_node_ids, batch_dst_node_ids))
        possible_random_edges = list(self.possible_edges - batch_edges)
        possible_random_edges = np.array([[src, dst] for src, dst in possible_random_edges], dtype=np.int64)
        possible_random_edges = possible_random_edges[np.lexsort((possible_random_edges[:, 1], possible_random_edges[:, 0]))]
        assert len(possible_random_edges) > 0
        # if replace is True, then a value in the list can be selected multiple times, otherwise, a value can be selected only once at most
        random_edge_indices = self.random_state.choice(len(possible_random_edges), size=size, replace=len(possible_random_edges) < size)
        return possible_random_edges[random_edge_indices][:, 0], possible_random_edges[random_edge_indices][:, 1]
        # return np.array([possible_random_edges[random_edge_idx][0] for random_edge_idx in random_edge_indices]), \
            #    np.array([possible_random_edges[random_edge_idx][1] for random_edge_idx in random_edge_indices])

    def historical_sample(self, size: int, batch_src_node_ids: np.ndarray, batch_dst_node_ids: np.ndarray,
                          current_batch_start_time: float, current_batch_end_time: float):
        """
        historical sampling strategy, first randomly samples among historical edges that are not in the current batch,
        if number of historical edges is smaller than size, then fill in remaining edges with randomly sampled edges
        :param size: int, number of sampled negative edges
        :param batch_src_node_ids: ndarray, shape (batch_size, ), source node ids in the current batch
        :param batch_dst_node_ids: ndarray, shape (batch_size, ), destination node ids in the current batch
        :param current_batch_start_time: float, start time in the current batch
        :param current_batch_end_time: float, end time in the current batch
        :return:
        """
        assert self.seed is not None
        mode = 'new'
        # get historical edges up to current_batch_start_time
        historical_edges = self.get_unique_edges_between_start_end_time(start_time=self.earliest_time, end_time=current_batch_start_time, mode=mode)
        # get edges in the current batch
        current_batch_edges = self.get_unique_edges_between_start_end_time(start_time=current_batch_start_time, end_time=current_batch_end_time, mode=mode)
        # get source and destination node ids of unique historical edges
        if mode == 'new':
            unique_historical_edges = get_diff_edges(historical_edges, current_batch_edges, self.max_nid)
            unique_historical_edges_src_node_ids = unique_historical_edges[:, 0]
            unique_historical_edges_dst_node_ids = unique_historical_edges[:, 1]
        else:
            unique_historical_edges = historical_edges - current_batch_edges
            unique_historical_edges_src_node_ids = np.array([edge[0] for edge in unique_historical_edges])
            unique_historical_edges_dst_node_ids = np.array([edge[1] for edge in unique_historical_edges])
            
        # if sample size is larger than number of unique historical edges, then fill in remaining edges with randomly sampled edges with collision check
        if size > len(unique_historical_edges):
            num_random_sample_edges = size - len(unique_historical_edges)
            random_sample_src_node_ids, random_sample_dst_node_ids = self.random_sample_with_collision_check(size=num_random_sample_edges,
             batch_src_node_ids=batch_src_node_ids,
             batch_dst_node_ids=batch_dst_node_ids,
             mode=mode)

            negative_src_node_ids = np.concatenate([random_sample_src_node_ids, unique_historical_edges_src_node_ids])
            negative_dst_node_ids = np.concatenate([random_sample_dst_node_ids, unique_historical_edges_dst_node_ids])
        else:
            historical_sample_edge_node_indices = self.random_state.choice(len(unique_historical_edges), size=size, replace=False)
            negative_src_node_ids = unique_historical_edges_src_node_ids[historical_sample_edge_node_indices]
            negative_dst_node_ids = unique_historical_edges_dst_node_ids[historical_sample_edge_node_indices]

        # Note that if one of the input of np.concatenate is empty, the output will be composed of floats.
        # Hence, convert the type to long to guarantee valid index
        return negative_src_node_ids.astype(np.int64), negative_dst_node_ids.astype(np.int64)

    def inductive_sample(self, size: int, batch_src_node_ids: np.ndarray, batch_dst_node_ids: np.ndarray,
                         current_batch_start_time: float, current_batch_end_time: float):
        """
        inductive sampling strategy, first randomly samples among inductive edges that are not in self.observed_edges and the current batch,
        if number of inductive edges is smaller than size, then fill in remaining edges with randomly sampled edges
        :param size: int, number of sampled negative edges
        :param batch_src_node_ids: ndarray, shape (batch_size, ), source node ids in the current batch
        :param batch_dst_node_ids: ndarray, shape (batch_size, ), destination node ids in the current batch
        :param current_batch_start_time: float, start time in the current batch
        :param current_batch_end_time: float, end time in the current batch
        :return:
        """
        assert self.seed is not None
        mode = 'new'
        # get historical edges up to current_batch_start_time
        historical_edges = self.get_unique_edges_between_start_end_time(start_time=self.earliest_time, end_time=current_batch_start_time, mode=mode)
        # get edges in the current batch
        current_batch_edges = self.get_unique_edges_between_start_end_time(start_time=current_batch_start_time, end_time=current_batch_end_time, mode=mode)
        # get source and destination node ids of historical edges but 1) not in self.observed_edges; 2) not in the current batch

        if mode == 'new':
            if len(self.observed_edges) == 0:
                self.observed_edges = np.zeros((0, 2), dtype=np.int64)
            elif type(self.observed_edges) != np.ndarray:
                self.observed_edges = np.array([[src, dst] for src, dst in self.observed_edges], dtype=np.int64)
            tmp_edges = get_diff_edges(historical_edges, self.observed_edges, self.max_nid)
            unique_inductive_edges = get_diff_edges(tmp_edges, current_batch_edges, self.max_nid)
            unique_inductive_edges_src_node_ids = unique_inductive_edges[:, 0]
            unique_inductive_edges_dst_node_ids = unique_inductive_edges[:, 1]
        else:
            unique_inductive_edges = historical_edges - self.observed_edges - current_batch_edges
            unique_inductive_edges_src_node_ids = np.array([edge[0] for edge in unique_inductive_edges])
            unique_inductive_edges_dst_node_ids = np.array([edge[1] for edge in unique_inductive_edges])
            
        # if sample size is larger than number of unique inductive edges, then fill in remaining edges with randomly sampled edges
        if size > len(unique_inductive_edges):
            num_random_sample_edges = size - len(unique_inductive_edges)
            random_sample_src_node_ids, random_sample_dst_node_ids = self.random_sample_with_collision_check(size=num_random_sample_edges,
             batch_src_node_ids=batch_src_node_ids,
             batch_dst_node_ids=batch_dst_node_ids,
             mode=mode)

            negative_src_node_ids = np.concatenate([random_sample_src_node_ids, unique_inductive_edges_src_node_ids])
            negative_dst_node_ids = np.concatenate([random_sample_dst_node_ids, unique_inductive_edges_dst_node_ids])
        else:
            inductive_sample_edge_node_indices = self.random_state.choice(len(unique_inductive_edges), size=size, replace=False)
            negative_src_node_ids = unique_inductive_edges_src_node_ids[inductive_sample_edge_node_indices]
            negative_dst_node_ids = unique_inductive_edges_dst_node_ids[inductive_sample_edge_node_indices]

        # Note that if one of the input of np.concatenate is empty, the output will be composed of floats.
        # Hence, convert the type to long to guarantee valid index
        return negative_src_node_ids.astype(np.int64), negative_dst_node_ids.astype(np.int64)

    def reset_random_state(self):
        """
        reset the random state by self.seed
        :return:
        """
        self.random_state = np.random.RandomState(self.seed)


def load_feat(d, rand_de=0, rand_dn=0):
    node_feats = None
    if os.path.exists('DATA/{}/node_features.pt'.format(d)):
        node_feats = torch.load('DATA/{}/node_features.pt'.format(d))
        if node_feats.dtype == torch.bool:
            node_feats = node_feats.type(torch.float32)
    edge_feats = None
    if os.path.exists('DATA/{}/edge_features.pt'.format(d)):
        edge_feats = torch.load('DATA/{}/edge_features.pt'.format(d))
        if edge_feats.dtype == torch.bool:
            edge_feats = edge_feats.type(torch.float32)
    if rand_de > 0:
        if d == 'LASTFM':
            edge_feats = torch.randn(1293103, rand_de)
        elif d == 'MOOC':
            edge_feats = torch.randn(411749, rand_de)
    if rand_dn > 0:
        if d == 'LASTFM':
            node_feats = torch.randn(1980, rand_dn)
        elif d == 'MOOC':
            node_feats = torch.randn(7144, rand_dn)
    return node_feats, edge_feats

def load_graph(d):
    df = pd.read_csv('DATA/{}/edges.csv'.format(d))
    g = np.load('DATA/{}/ext_full.npz'.format(d))
    return g, df

def parse_config(f):
    conf = yaml.safe_load(open(f, 'r'))
    sample_param = conf['sampling'][0]
    memory_param = conf['memory'][0]
    gnn_param = conf['gnn'][0]
    train_param = conf['train'][0]
    return sample_param, memory_param, gnn_param, train_param

def to_dgl_blocks(ret, hist, reverse=False, cuda=True):
    mfgs = list()
    for r in ret:
        if not reverse:
            b = dgl.create_block((r.col(), r.row()), num_src_nodes=r.dim_in(), num_dst_nodes=r.dim_out())
            b.srcdata['ID'] = torch.from_numpy(r.nodes())
            b.edata['dt'] = torch.from_numpy(r.dts())[b.num_dst_nodes():]
            b.srcdata['ts'] = torch.from_numpy(r.ts())
        else:
            b = dgl.create_block((r.row(), r.col()), num_src_nodes=r.dim_out(), num_dst_nodes=r.dim_in())
            b.dstdata['ID'] = torch.from_numpy(r.nodes())
            b.edata['dt'] = torch.from_numpy(r.dts())[b.num_src_nodes():]
            b.dstdata['ts'] = torch.from_numpy(r.ts())
        b.edata['ID'] = torch.from_numpy(r.eid())
        if cuda:
            mfgs.append(b.to('cuda:0'))
        else:
            mfgs.append(b)
    mfgs = list(map(list, zip(*[iter(mfgs)] * hist)))
    mfgs.reverse()
    return mfgs

def node_to_dgl_blocks(root_nodes, ts, cuda=True):
    mfgs = list()
    b = dgl.create_block(([],[]), num_src_nodes=root_nodes.shape[0], num_dst_nodes=root_nodes.shape[0])
    b.srcdata['ID'] = torch.from_numpy(root_nodes)
    b.srcdata['ts'] = torch.from_numpy(ts)
    if cuda:
        mfgs.insert(0, [b.to('cuda:0')])
    else:
        mfgs.insert(0, [b])
    return mfgs

def mfgs_to_cuda(mfgs):
    for mfg in mfgs:
        for i in range(len(mfg)):
            mfg[i] = mfg[i].to('cuda:0')
    return mfgs

def prepare_input(mfgs, node_feats, edge_feats, combine_first=False, pinned=False, nfeat_buffs=None, efeat_buffs=None, nids=None, eids=None):
    if combine_first:
        for i in range(len(mfgs[0])):
            if mfgs[0][i].num_src_nodes() > mfgs[0][i].num_dst_nodes():
                num_dst = mfgs[0][i].num_dst_nodes()
                ts = mfgs[0][i].srcdata['ts'][num_dst:]
                nid = mfgs[0][i].srcdata['ID'][num_dst:].float()
                nts = torch.stack([ts, nid], dim=1)
                unts, idx = torch.unique(nts, dim=0, return_inverse=True)
                uts = unts[:, 0]
                unid = unts[:, 1]
                # import pdb; pdb.set_trace()
                b = dgl.create_block((idx + num_dst, mfgs[0][i].edges()[1]), num_src_nodes=unts.shape[0] + num_dst, num_dst_nodes=num_dst, device=torch.device('cuda:0'))
                b.srcdata['ts'] = torch.cat([mfgs[0][i].srcdata['ts'][:num_dst], uts], dim=0)
                b.srcdata['ID'] = torch.cat([mfgs[0][i].srcdata['ID'][:num_dst], unid], dim=0)
                b.edata['dt'] = mfgs[0][i].edata['dt']
                b.edata['ID'] = mfgs[0][i].edata['ID']
                mfgs[0][i] = b
    t_idx = 0
    t_cuda = 0
    i = 0
    if node_feats is not None:
        for b in mfgs[0]:
            if pinned:
                if nids is not None:
                    idx = nids[i]
                else:
                    idx = b.srcdata['ID'].cpu().long()
                torch.index_select(node_feats, 0, idx, out=nfeat_buffs[i][:idx.shape[0]])
                b.srcdata['h'] = nfeat_buffs[i][:idx.shape[0]].cuda(non_blocking=True)
                i += 1
            else:
                srch = node_feats[b.srcdata['ID'].cpu().long()].float()
                b.srcdata['h'] = srch.cuda()
    i = 0
    if edge_feats is not None:
        for mfg in mfgs:
            for b in mfg:
                if b.num_src_nodes() > b.num_dst_nodes():
                    if pinned:
                        if eids is not None:
                            idx = eids[i]
                        else:
                            idx = b.edata['ID'].cpu().long()
                        torch.index_select(edge_feats, 0, idx, out=efeat_buffs[i][:idx.shape[0]])
                        b.edata['f'] = efeat_buffs[i][:idx.shape[0]].cuda(non_blocking=True)
                        i += 1
                    else:
                        srch = edge_feats[b.edata['ID'].cpu().long()].float()
                        b.edata['f'] = srch.cuda()
    return mfgs

def get_ids(mfgs, node_feats, edge_feats):
    nids = list()
    eids = list()
    if node_feats is not None:
        for b in mfgs[0]:
            nids.append(b.srcdata['ID'].cpu().long())
    if 'ID' in mfgs[0][0].edata:
        if edge_feats is not None:
            for mfg in mfgs:
                for b in mfg:
                    eids.append(b.edata['ID'].cpu().long())
    else:
        eids = None
    return nids, eids

def get_pinned_buffers(sample_param, batch_size, node_feats, edge_feats):
    pinned_nfeat_buffs = list()
    pinned_efeat_buffs = list()
    limit = int(batch_size * 3.3)
    if 'neighbor' in sample_param:
        for i in sample_param['neighbor']:
            limit *= i + 1
            if edge_feats is not None:
                for _ in range(sample_param['history']):
                    pinned_efeat_buffs.insert(0, torch.zeros((limit, edge_feats.shape[1]), pin_memory=True))
    if node_feats is not None:
        for _ in range(sample_param['history']):
            pinned_nfeat_buffs.insert(0, torch.zeros((limit, node_feats.shape[1]), pin_memory=True))
    return pinned_nfeat_buffs, pinned_efeat_buffs

class NeighborSampler:

    def __init__(self, adj_list: list, dst_node_ids: np.ndarray, sample_neighbor_strategy: str = 'uniform', time_scaling_factor: float = 0.0, seed: int = None, unique: bool=True, add_freq: bool=False):
        """
        Neighbor sampler.
        :param adj_list: list, list of list, where each element is a list of triple tuple (node_id, edge_id, timestamp)
        :param sample_neighbor_strategy: str, how to sample historical neighbors, 'uniform', 'recent', or 'time_interval_aware'
        :param time_scaling_factor: float, a hyper-parameter that controls the sampling preference with time interval,
        a large time_scaling_factor tends to sample more on recent links, this parameter works when sample_neighbor_strategy == 'time_interval_aware'
        :param seed: int, random seed
        """
        self.sample_neighbor_strategy = sample_neighbor_strategy
        self.seed = seed
        self.unique_dst_node_ids = np.unique(dst_node_ids).astype(np.int64)

        # list of each node's neighbor ids, edge ids and interaction times, which are sorted by interaction times
        self.nodes_neighbor_ids = []
        self.nodes_neighbor_times = []

        if self.sample_neighbor_strategy == 'time_interval_aware':
            self.nodes_neighbor_sampled_probabilities = []
            self.time_scaling_factor = time_scaling_factor

        # the list at the first position in adj_list is empty, hence, sorted() will return an empty list for the first position
        # its corresponding value in self.nodes_neighbor_ids, self.nodes_neighbor_times will also be empty with length 0
        for node_idx, per_node_neighbors in enumerate(adj_list):
            # per_node_neighbors is an array [neighbor_id, edge_id, timestamp]
            # sort the list based on timestamps, sorted() function is stable
            # Note that sort the list based on edge id is also correct, as the original data file ensures the interactions are chronological
            if len(per_node_neighbors) == 0:
                self.nodes_neighbor_ids.append(np.array([]))
                self.nodes_neighbor_times.append(np.array([]))
                if self.sample_neighbor_strategy == 'time_interval_aware':
                    self.nodes_neighbor_sampled_probabilities.append(np.array([]))
                continue

            if unique:
                # extracts unique neighbors ids and latest times
                sorted_per_node_neighbors = per_node_neighbors[np.argsort(per_node_neighbors[:, 0])]
                node_neighbor_ids, start_indices = np.unique(sorted_per_node_neighbors[:, 0], return_index=True)
                nodes_neighbor_times = np.maximum.reduceat(sorted_per_node_neighbors[:, 2], start_indices)
                sorted_idx = np.argsort(nodes_neighbor_times)
                self.nodes_neighbor_ids.append(node_neighbor_ids[sorted_idx])
                self.nodes_neighbor_times.append(nodes_neighbor_times[sorted_idx])
                
                # additional for time interval aware sampling strategy (proposed in CAWN paper)
                if self.sample_neighbor_strategy == 'time_interval_aware':
                    nodes_neighbor_sampled_probabilities = self.compute_sampled_probabilities(nodes_neighbor_times[sorted_idx])
                    if add_freq:
                        counts = np.diff(np.append(start_indices, len(sorted_per_node_neighbors))) # Count the number of items for each node_neighbor_id
                        nodes_neighbor_sampled_probabilities *= 1/(1+np.exp(-counts[sorted_idx]))
                    self.nodes_neighbor_sampled_probabilities.append(nodes_neighbor_sampled_probabilities)

            else:
                sorted_per_node_neighbors = per_node_neighbors[np.argsort(per_node_neighbors[:, 2])]
                self.nodes_neighbor_ids.append(sorted_per_node_neighbors[:, 0])
                self.nodes_neighbor_times.append(sorted_per_node_neighbors[:, 2])

                # additional for time interval aware sampling strategy (proposed in CAWN paper)
                if self.sample_neighbor_strategy == 'time_interval_aware':
                    self.nodes_neighbor_sampled_probabilities.append(self.compute_sampled_probabilities(sorted_per_node_neighbors[:, 2]))

        if self.seed is not None:
            self.random_state = np.random.RandomState(self.seed)

    def random_sample(self, size: int, dst_id: int):
        """
        Random sampling strategy, which is used by previous works.
        :param size: int, number of sampled negative edges.
        :param dst_id: int, the destination node id that should not be in the sampled result.
        :return: array of sampled negative edges excluding dst_id.
        """
        if self.seed is None:
            random_sample_edge_dst_node_indices = np.random.randint(0, len(self.unique_dst_node_ids), size)
        else:
            random_sample_edge_dst_node_indices = self.random_state.randint(0, len(self.unique_dst_node_ids), size)
        return self.unique_dst_node_ids[random_sample_edge_dst_node_indices]
    
    def compute_sampled_probabilities(self, node_neighbor_times: np.ndarray):
        """
        compute the sampled probabilities of historical neighbors based on their interaction times
        :param node_neighbor_times: ndarray, shape (num_historical_neighbors, )
        :return:
        """
        if len(node_neighbor_times) == 0:
            return np.array([])
        # compute the time delta with regard to the last time in node_neighbor_times
        node_neighbor_times = node_neighbor_times - np.max(node_neighbor_times)
        # compute the normalized sampled probabilities of historical neighbors
        exp_node_neighbor_times = np.exp(self.time_scaling_factor * node_neighbor_times)
        sampled_probabilities = exp_node_neighbor_times / np.cumsum(exp_node_neighbor_times)
        # note that the first few values in exp_node_neighbor_times may be all zero, which make the corresponding values in sampled_probabilities
        # become nan (divided by zero), so we replace the nan by a very large negative number -1e10 to denote the sampled probabilities
        sampled_probabilities[np.isnan(sampled_probabilities)] = -1e10
        return sampled_probabilities

    def find_neighbors_before(self, node_id: int, interact_time: float, return_sampled_probabilities: bool = False):
        """
        extracts all the interactions happening before interact_time (less than interact_time) for node_id in the overall interaction graph
        the returned interactions are sorted by time.
        :param node_id: int, node id
        :param interact_time: float, interaction time
        :param return_sampled_probabilities: boolean, whether return the sampled probabilities of neighbors
        :return: neighbors, edge_ids, timestamps and sampled_probabilities (if return_sampled_probabilities is True) with shape (historical_nodes_num, )
        """
        # return index i, which satisfies list[i - 1] < v <= list[i]
        # return 0 for the first position in self.nodes_neighbor_times since the value at the first position is empty
        i = np.searchsorted(self.nodes_neighbor_times[node_id], interact_time)

        if return_sampled_probabilities:
            return self.nodes_neighbor_ids[node_id][:i], self.nodes_neighbor_times[node_id][:i], self.nodes_neighbor_sampled_probabilities[node_id][:i]
        else:
            return self.nodes_neighbor_ids[node_id][:i], self.nodes_neighbor_times[node_id][:i], None

    def sampler(self, node_ids: np.ndarray, dst_ids: np.ndarray, node_interact_times: np.ndarray, num_neighbors: int = 20, num_random: int = 20):
        """
        get historical neighbors of nodes in node_ids with interactions before the corresponding time in node_interact_times
        :param node_ids: ndarray, shape (batch_size, ) or (*, ), node ids
        :param node_interact_times: ndarray, shape (batch_size, ) or (*, ), node interaction times
        :param num_neighbors: int, number of neighbors to sample for each node
        :return:
        """
        # All interactions described in the following three matrices are sorted in each row by time
        # each entry in position (i,j) represents the id of the j-th dst node of src node node_ids[i] with an interaction before node_interact_times[i]
        # ndarray, shape (batch_size, num_neighbors)
        nodes_neighbor_ids = np.zeros((len(node_ids), num_neighbors+num_random)).astype(np.longlong)
        # each entry in position (i,j) represents the interaction time between src node node_ids[i] and dst node nodes_neighbor_ids[i][j], before node_interact_times[i]
        # ndarray, shape (batch_size, num_neighbors)
        nodes_neighbor_times = np.zeros((len(node_ids), num_neighbors+num_random)).astype(np.float32)

        # extracts all neighbors ids, edge ids and interaction times of nodes in node_ids, which happened before the corresponding time in node_interact_times
        for idx, (node_id, dst_id, node_interact_time) in enumerate(zip(node_ids, dst_ids, node_interact_times)):
            # find neighbors that interacted with node_id before time node_interact_time
            node_neighbor_ids, node_neighbor_times, node_neighbor_sampled_probabilities = \
                self.find_neighbors_before(node_id=node_id, interact_time=node_interact_time, return_sampled_probabilities=self.sample_neighbor_strategy == 'time_interval_aware')
            
            # remove positive
            true_mask = node_neighbor_ids!=dst_id
            node_neighbor_ids = node_neighbor_ids[true_mask]
            node_neighbor_times = node_neighbor_times[true_mask]
            if node_neighbor_sampled_probabilities is not None:
                node_neighbor_sampled_probabilities = node_neighbor_sampled_probabilities[true_mask]

            len_neighbors = min(len(node_neighbor_ids), num_neighbors)
            if len_neighbors > 0:
                if self.sample_neighbor_strategy in ['uniform', 'time_interval_aware']:
                    # when self.sample_neighbor_strategy == 'uniform', we shuffle the data before sampling with node_neighbor_sampled_probabilities as None
                    # when self.sample_neighbor_strategy == 'time_interval_aware', we sample neighbors based on node_neighbor_sampled_probabilities
                    # for time_interval_aware sampling strategy, we additionally use softmax to make the sum of sampled probabilities be 1
                    if node_neighbor_sampled_probabilities is not None:
                        # for extreme case that node_neighbor_sampled_probabilities only contains -1e10, which will make the denominator of softmax be zero,
                        # torch.softmax() function can tackle this case
                        node_neighbor_sampled_probabilities = torch.softmax(torch.from_numpy(node_neighbor_sampled_probabilities).float(), dim=0).numpy()
                    if self.seed is None:
                        sampled_indices = np.random.choice(a=len(node_neighbor_ids), size=len_neighbors, p=node_neighbor_sampled_probabilities)
                    else:
                        sampled_indices = self.random_state.choice(a=len(node_neighbor_ids), size=len_neighbors, p=node_neighbor_sampled_probabilities)

                    nodes_neighbor_ids[idx, -len_neighbors:] = node_neighbor_ids[sampled_indices]
                    nodes_neighbor_times[idx, -len_neighbors:] = node_neighbor_times[sampled_indices]

                    # resort based on timestamps, return the ids in sorted increasing order, note this maybe unstable when multiple edges happen at the same time
                    # (we still do this though this is unnecessary for TGAT or CAWN to guarantee the order of nodes,
                    # since TGAT computes in an order-agnostic manner with relative time encoding, and CAWN computes for each walk while the sampled nodes are in different walks)
                    sorted_position = nodes_neighbor_times[idx, -len_neighbors:].argsort()
                    nodes_neighbor_ids[idx, -len_neighbors:] = nodes_neighbor_ids[idx, -len_neighbors:][sorted_position]
                    nodes_neighbor_times[idx, -len_neighbors:] = nodes_neighbor_times[idx, -len_neighbors:][sorted_position]
                elif self.sample_neighbor_strategy == 'recent':
                    # Take most recent interactions with number num_neighbors
                    # put the neighbors' information at the back positions
                    nodes_neighbor_ids[idx, -len_neighbors:] = node_neighbor_ids[-len_neighbors:]
                    nodes_neighbor_times[idx, -len_neighbors:] = node_neighbor_times[-len_neighbors:]
                else:
                    raise ValueError(f'Not implemented error for sample_neighbor_strategy {self.sample_neighbor_strategy}!')
                
            num_random_new = num_random + max(num_neighbors - len_neighbors, 0)
            nodes_neighbor_ids[idx, :num_random_new] = self.random_sample(num_random_new, dst_id)

        # two ndarrays, with shape (batch_size, num_neighbors)
        return nodes_neighbor_ids, nodes_neighbor_times

def get_neighbor_sampler(data, sample_neighbor_strategy: str = 'uniform', time_scaling_factor: float = 0.0, seed: int = None, unique: bool=True, add_freq: bool=False):
#def get_neighbor_sampler(data, sample_neighbor_strategy: str = 'uniform', time_scaling_factor: float = 0.0, seed: int = None, unique: bool=True, add_freq: bool=False, bipartite: bool = True):
    """
    get neighbor sampler
    :param data: pd.dataframe
    :param sample_neighbor_strategy: str, how to sample historical neighbors, 'uniform', 'recent', or 'time_interval_aware''
    :param time_scaling_factor: float, a hyper-parameter that controls the sampling preference with time interval,
    a large time_scaling_factor tends to sample more on recent links, this parameter works when sample_neighbor_strategy == 'time_interval_aware'
    :param seed: int, random seed
    :return:
    """
    max_node_id = max(data.src.values.max(), data.dst.values.max())
    # the adjacency vector stores edges for each node (source or destination), undirected
    # adj_list, list of list, where each element is a list of triple tuple (node_id, edge_id, timestamp)
    # the list at the first position in adj_list is empty
    adj_list = [[] for _ in range(max_node_id + 1)]
    for src_node_id, dst_node_id, edge_id, node_interact_time in zip(data.src.values, data.dst.values, data.eid.values, data.time.values):
        adj_list[src_node_id].append([dst_node_id, edge_id, node_interact_time])
        adj_list[dst_node_id].append([src_node_id, edge_id, node_interact_time])
    adj_list = [np.array(sublist) for sublist in adj_list]

    '''if bipartite:
        dst_node_ids=data.dst.values
    else:
        dst_node_ids=np.concatenate((data.src.values, data.dst.values))
    '''
    dst_node_ids=data.dst.values
    return NeighborSampler(adj_list=adj_list, dst_node_ids=dst_node_ids, sample_neighbor_strategy=sample_neighbor_strategy, time_scaling_factor=time_scaling_factor, seed=seed, unique=unique, add_freq=add_freq)


if __name__ == '__main__':
    import argparse
    import copy
    import os
    import logging
    import socket
    from tqdm import tqdm
    from data_adaptor import load_data

    parser=argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='dataset name')
    args=parser.parse_args()
    args.hostname = socket.gethostname()

    set_logger(f'{args.data}', '', log_file=False)
    set_random_seed(42)
    logging.info('''We rewrite random_sample_with_collision_check() of NegativeEdgeSampler to avoid OOM caused by self.possible edges. Firstly we generate the sampled edge indices and then iteratively get the corresponding (src, dst) pair, which presents about 1.5 times slower speed. Secondly we speedup the get_unique_edges_between_start_end_time and the difference set between two sets of edges by remapping the (src, dst) pair into the edge idx with np.int64. Finally, we use numba.jit(nopython=True, nogil=True) to speedup.''')

    logging.info(__file__)
    logging.info(args)
    logging.info('Loading Dataset %s.', args.data)
    train_g, full_g, df, train_df, val_df, test_df, new_val_df, new_test_df, nfeat, efeat = load_data(args.data)
    df, train_df, val_df, test_df, new_val_df, new_test_df = [tmp.reset_index(drop=True) for tmp in [df, train_df, val_df, test_df, new_val_df, new_test_df]]
    df['eid'] = np.arange(len(df))

    # The same as DyGLib https://github.com/yule-BUAA/DyGLib/train_link_prediction.py.
    # initialize negative samplers, set seeds for validation and testing so negatives are the same across different runs
    # in the inductive setting, negatives are sampled only amongst other new nodes
    # train negative edge sampler does not need to specify the seed, but evaluation samplers need to do so
    train_neg_sampler = NegativeEdgeSampler(src_node_ids=train_df['src'], dst_node_ids=train_df['dst'])
    val_neg_sampler = NegativeEdgeSampler(src_node_ids=df['src'], dst_node_ids=df['dst'], seed=0)
    nval_neg_sampler = NegativeEdgeSampler(src_node_ids=new_val_df['src'], dst_node_ids=new_val_df['dst'], seed=1)
    test_neg_sampler = NegativeEdgeSampler(src_node_ids=df['src'], dst_node_ids=df['dst'], seed=2)
    ntest_neg_sampler = NegativeEdgeSampler(src_node_ids=new_test_df['src'], dst_node_ids=new_test_df['dst'], seed=3)
    
    neg_samples = 1
    def check_sampler(eval_df, sampler, mode='random'):
        # We check whether the behaviors of mode="old" and mode="new" are the same.
        # We re-write possible_random_edges in NegativeEdgeSampler.random_sample_with_collision_check
        # to make it consistent with our sorted edge indices, where the original implementation is 
        # randomly sorted with the set() class.
        new_time, old_time = 0.0, 0.0
        for i, rows in tqdm(eval_df.groupby(eval_df.index // 200)):
            src, dst, ts = rows.src.values, rows.dst.values, rows.time.values
            src, dst = src.astype(np.int64), dst.astype(np.int64)
            
            # Follow DyGLib.
            sampler.seed = i
            sampler.reset_random_state()
            new_time_s = time.time()
            nsrc, ndst = sampler.random_sample_with_collision_check(len(src), src, dst, mode='new')
            new_time += time.time() - new_time_s

            old_time_s = time.time()
            sampler.reset_random_state()
            osrc, odst = sampler.random_sample_with_collision_check(len(src), src, dst, mode='old')
            old_time += time.time() - old_time_s

            assert np.all(nsrc == osrc) and np.all(ndst == odst)

            sampler.reset_random_state()
            nedges = sampler.get_unique_edges_between_start_end_time(ts[0], ts[-1], mode='new')
            sampler.reset_random_state()
            oedges = sampler.get_unique_edges_between_start_end_time(ts[0], ts[-1], mode='old')
            oedges = np.array([[src, dst] for src, dst in oedges], dtype=np.int64)
            oedges = oedges[np.lexsort((oedges[:, 1], oedges[:, 0]))]
            assert np.all(nedges == oedges)

        logging.info('New version: %.2f seconds, Old version: %.2f seconds, Speedup: %.0f times.', 
                     new_time, old_time, old_time / new_time)
        logging.info('Check successfully.')
        return 0.0, 0.0

    # Follow DGB and DyGLib.
    logging.info('********** Historical negative testing. **********')
    val_params = {'src_node_ids': df['src'], 'dst_node_ids': df['dst'], 'interact_times': df['time'], 'last_observed_time': train_df['time'].iloc[-1], 'negative_sample_strategy':'historical', 'seed': 0}
    nval_params = {'src_node_ids': new_val_df['src'], 'dst_node_ids': new_val_df['dst'], 'interact_times': new_val_df['time'], 'last_observed_time': train_df['time'].iloc[-1], 'negative_sample_strategy':'historical', 'seed': 1}
    test_params = {'src_node_ids': df['src'], 'dst_node_ids': df['dst'], 'interact_times': df['time'], 'last_observed_time': val_df['time'].iloc[-1], 'negative_sample_strategy':'historical', 'seed': 2}
    ntest_params = {'src_node_ids': new_test_df['src'], 'dst_node_ids': new_test_df['dst'], 'interact_times': new_test_df['time'], 'last_observed_time': val_df['time'].iloc[-1], 'negative_sample_strategy':'historical', 'seed': 3}

    check_sampler(val_df, NegativeEdgeSampler(**val_params), mode='historical')
    check_sampler(new_val_df, NegativeEdgeSampler(**nval_params), mode='historical')
    check_sampler(test_df, NegativeEdgeSampler(**test_params), mode='historical')
    check_sampler(new_test_df, NegativeEdgeSampler(**ntest_params), mode='historical')

    logging.info('********** Inductive negative testing. **********')
    val_params['negative_sample_strategy'] = 'inductive'
    nval_params['negative_sample_strategy'] = 'inductive'
    test_params['negative_sample_strategy'] = 'inductive'
    ntest_params['negative_sample_strategy'] = 'inductive'

    check_sampler(val_df, NegativeEdgeSampler(**val_params), mode='inductive')
    check_sampler(new_val_df, NegativeEdgeSampler(**nval_params), mode='inductive')
    check_sampler(test_df, NegativeEdgeSampler(**test_params), mode='inductive')
    check_sampler(new_test_df, NegativeEdgeSampler(**ntest_params), mode='inductive')
