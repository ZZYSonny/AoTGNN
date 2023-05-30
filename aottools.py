import warnings
from typing import *

import torch

from torch_geometric.data import Dataset


class GnmSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset: Dataset, max_num_graphs: int, max_num_nodes: int, max_num_edges: int, shuffle: bool):
        self.dataset = dataset
        self.max_num_graphs = max_num_graphs
        self.max_num_nodes = max_num_nodes
        self.max_num_edges = max_num_edges
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[List[int]]:
        batch = []
        num_processed = 0
        cur_num_graphs = 0
        cur_num_nodes = 0
        cur_num_edges = 0

        if self.shuffle: indices = torch.randperm(len(self.dataset), dtype=torch.long)
        else: indices = torch.arange(len(self.dataset), dtype=torch.long)

        while (num_processed < len(self.dataset)):
            # Fill batch
            for idx in indices[num_processed:]:
                # Size of sample
                data = self.dataset[idx]
                cur_num_graphs += 1
                cur_num_nodes += data.num_nodes
                cur_num_edges += data.num_edges

                if cur_num_graphs + 1 > self.max_num_graphs \
                    or cur_num_nodes > self.max_num_nodes \
                    or cur_num_edges > self.max_num_edges:
                    break

                # Add sample to current batch
                batch.append(idx.item())
                num_processed += 1

            yield batch
            batch = []
            cur_num_graphs = 0
            cur_num_nodes = 0
            cur_num_edges = 0

    def __len__(self) -> int:
        return 99999


from collections.abc import Mapping

import torch.utils.data
from torch_geometric.data import Batch, Dataset
from torch_geometric.data.data import Data, BaseData
from torch_geometric.data.datapipes import DatasetAdapter

class GnmCollater:
    def __init__(self, max_num_graphs: int, max_num_nodes: int, max_num_edges: int):
        self.max_num_graphs = max_num_graphs
        self.max_num_nodes = max_num_nodes
        self.max_num_edges = max_num_edges

    def fake_graph(self, num_nodes, num_edges, data0):
        return Data(
                x = torch.zeros(
                    [num_nodes] + list(data0.x.shape[1:]), 
                    dtype=data0.x.dtype, 
                    device=data0.x.device
                ),
                y = torch.randn_like(data0.y),
                edge_index = torch.full(
                    [2, num_edges], 
                    0, 
                    dtype=torch.long, 
                    device=data0.edge_index.device
                ),
                edge_attr = None if data0.edge_attr is None else torch.zeros(
                    [num_edges] + list(data0.edge_attr.shape[1:]), 
                    dtype=data0.edge_attr.dtype, 
                    device=data0.edge_attr.device
                ),
            )

    def __call__(self, batch: List[BaseData]):
        data0 = batch[0]
        assert(isinstance(data0, BaseData))
        
        num_graphs = len(batch)
        num_nodes = sum([data.num_nodes for data in batch])
        num_edges = sum([data.num_edges for data in batch])
        if self.max_num_graphs - num_graphs - 1 > 0:
            extra_data = [self.fake_graph(0, 0, data0)] * (self.max_num_graphs - num_graphs - 1)
        else:
            extra_data = []
        extra_data.append(self.fake_graph(self.max_num_nodes - num_nodes, self.max_num_edges - num_edges, data0))

        return Batch.from_data_list(batch + extra_data)

def pad_num(x: int, d: int):
    return int(x//d+1)*d;

class GnmLoader(torch.utils.data.DataLoader):
    def __init__(
        self,
        dataset: Union[Dataset, Sequence[BaseData], DatasetAdapter],
        batch_size: int,
        shuffle: bool = False,
        **kwargs,
    ):
        avg_num_nodes = sum([data.num_nodes for data in dataset]) / len(dataset)
        avg_num_edges = sum([data.num_edges for data in dataset]) / len(dataset)
        max_num_nodes = pad_num(int(avg_num_nodes * (batch_size-1)),16)
        max_num_edges = pad_num(int(avg_num_edges * (batch_size-1)),16)
        super().__init__(
            dataset,
            batch_sampler=GnmSampler(dataset, batch_size-1, max_num_nodes-1, max_num_edges, shuffle),
            collate_fn=GnmCollater(batch_size, max_num_nodes, max_num_edges),
            **kwargs,
        )

import os
import os.path as osp
from tqdm import tqdm
class PreBatchedDataset():
    def __init__(self,path:str,loader:torch.utils.data.DataLoader,max_epochs:int):
        self.path = path
        self.cnt = 0
        self.max_epochs = max_epochs

        if not osp.exists(path):
            os.mkdir(path)
            for i in tqdm(range(max_epochs)):
                datalist = list(loader)
                with open(osp.join(path,f'{i}.pt.xz'),'wb') as f:
                    torch.save(datalist,f)
    
    def __iter__(self):
        with open(osp.join(self.path,f'{self.cnt}.pt.xz'),'rb') as f:
            datalist = torch.load(f)
        if self.cnt < self.max_epochs:
            self.cnt += 1
        else:
            self.cnt = 0
        return iter(datalist)