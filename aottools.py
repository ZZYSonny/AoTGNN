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
                    or cur_num_nodes + (self.max_num_graphs - cur_num_graphs) > self.max_num_nodes \
                    or cur_num_edges + (self.max_num_graphs - cur_num_graphs) > self.max_num_edges:
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

        extra_data = []
        for _ in range(num_graphs, self.max_num_graphs-1):
            extra_data.append(self.fake_graph(1, 1, data0))
            num_nodes += 1
            num_edges += 1
        
        extra_data.append(self.fake_graph(self.max_num_nodes - num_nodes, self.max_num_edges - num_edges, data0))

        return Batch.from_data_list(batch + extra_data)

class GnmLoader(torch.utils.data.DataLoader):
    def __init__(
        self,
        dataset: Union[Dataset, Sequence[BaseData], DatasetAdapter],
        batch_size: int,
        shuffle: bool = False,
        **kwargs,
    ):
        #avg_num_nodes = 24
        #avg_num_edges = 50
        avg_num_nodes = sum([data.num_nodes for data in dataset]) / len(dataset)
        avg_num_edges = sum([data.num_edges for data in dataset]) / len(dataset)
        max_num_nodes = int(avg_num_nodes * batch_size)
        max_num_edges = int(avg_num_edges * batch_size)
        super().__init__(
            dataset,
            batch_sampler=GnmSampler(dataset, batch_size+1, max_num_nodes, max_num_edges, shuffle),
            collate_fn=GnmCollater(batch_size+1, max_num_nodes, max_num_edges),
            **kwargs,
        )

from contextlib import nullcontext
from functools import partial
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader


class PrefetchLoader:
    r"""A GPU prefetcher class for asynchronously transferring data of a
    :class:`torch.utils.data.DataLoader` from host memory to device memory.

    Args:
        loader (torch.utils.data.DataLoader): The data loader.
        device (torch.device, optional): The device to load the data to.
            (default: :obj:`None`)
    """
    def __init__(
        self,
        loader: DataLoader,
        device: Optional[torch.device] = None,
    ):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.loader = loader
        self.device = torch.device(device)

        self.is_cuda = torch.cuda.is_available() and self.device.type == 'cuda'

    def non_blocking_transfer(self, batch: Any) -> Any:
        if not self.is_cuda:
            return batch
        if isinstance(batch, (list, tuple)):
            return [self.non_blocking_transfer(v) for v in batch]
        if isinstance(batch, dict):
            return {k: self.non_blocking_transfer(v) for k, v in batch.items()}

        batch = batch.pin_memory()
        return batch.to(self.device, non_blocking=True)

    def __iter__(self) -> Any:
        first = True
        if self.is_cuda:
            stream = torch.cuda.Stream()
            stream_context = partial(torch.cuda.stream, stream=stream)
        else:
            stream = None
            stream_context = nullcontext

        for next_batch in self.loader:

            with stream_context():
                next_batch = self.non_blocking_transfer(next_batch)

            if not first:
                yield batch  # noqa
            else:
                first = False

            if stream is not None:
                torch.cuda.current_stream().wait_stream(stream)

            batch = next_batch

        yield batch

    def __len__(self) -> int:
        return len(self.loader)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.loader})'