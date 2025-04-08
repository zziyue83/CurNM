# Curriculum Negative Mining For Temporal Networks

## Overview

This repo is the open-sourced code for our work *Curriculum Negative Mining For Temporal Networks*.
This repository contains the open-source implementation of our work, *Curriculum Negative Mining for Temporal Networks*. It builds upon the frameworks of [TGL](https://github.com/amazon-science/tgl) and [DyGLib](https://github.com/yule-BUAA/DyGLib).

## Requirements
- python >= 3.6.13
- pytorch >= 1.8.1
- pandas >= 1.1.5
- numpy >= 1.19.5
- dgl >= 0.6.1
- pyyaml >= 5.4.1
- tqdm >= 4.61.0
- pybind11 >= 2.6.2
- g++ >= 7.5.0
- openmp >= 201511

## Dataset

We adopt the dataset format used by TGL. All dataset files should be placed in the `./DATA/` directory.

1. `edges.csv`: This file contains the temporal edge information of the graph. 
The CSV file must include the following columns with the header: `,src,dst,time,ext_roll` where each column represents the edge index (starting from 0), source node index (starting from 0), destination node index, timestamp, and extrapolation roll (0 for training edges, 1 for validation edges, and 2 for test edges). The rows should be sorted in ascending order by the time column.
2. `ext_full.npz`: This file contains the T-CSR representation of the temporal graph. You can generate it from `edges.csv` using the provided script with the following command:
    >python gen_graph.py --data \<NameOfYourDataset>
3. `edge_features.pt` (optional): A PyTorch tensor that stores edge features in a row-wise format with shape (num_edges, edge_feature_dim). *Note: At least one of `edge_features.pt` or `node_features.pt` must be present.*
4. `node_features.pt` (optional): The torch tensor that stores the node featrues row-wise with shape (num nodes, dim node features). *Note: at least one of `edge_features.pt` or `node_features.pt` should present.*
5. `labels.csv` (optional): This file contains node labels for the dynamic node classification task. It must include the following columns with the header:`,node,time,label,ext_roll` where each column represents node label index (start with 0), node index (start with 0), timestamp, node label, extrapolation roll (0 for training node labels, 1 for validation node labels, 2 for test node labels). The rows should be sorted in ascending order by the time column.

## Run

python main.py --config config/<NameOfYourModel>.yml --model_name <NameOfYourModel> --data <NameOfYourDataset>
