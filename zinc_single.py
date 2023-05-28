import warnings
warnings.filterwarnings("ignore")

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import torch_geometric as tg
import torch_geometric.nn as tgnn
import torch_geometric.loader as tgloader
import torch_geometric.datasets as tgsets

parser = argparse.ArgumentParser()
parser.add_argument("--aot", action="store_true")
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--loader_workers", type=int, default=4)
parser.add_argument("--batch_repeats", type=int, default=1000)
parser.add_argument("--epoch_repeats", type=int, default=10)
args = parser.parse_args()
print(args)


class ModelGraph(nn.Module):
    def __init__(self, dim: int, num_layer: int):
        super().__init__()
        self.embed = nn.Embedding(22, dim)
        self.gnn = tgnn.GIN(dim, dim, 5, dim)
        self.agg = tgnn.DeepSetsAggregation(nn.Identity(), nn.Linear(dim, 1))

    def forward(self, batch):
        h = self.embed(batch.x.flatten())
        h = self.gnn(h, batch.edge_index)
        return self.agg(h, batch.batch)
        

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = tgsets.ZINC(".dataset", True, split="train")
loader = tgloader.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.loader_workers)
data = next(iter(loader)).to(device)

model = ModelGraph(128, 3).to(device)
if args.aot: model = tg.compile(model, mode="max-autotune")

for _ in tqdm(range(1), desc="Warm"):
    for _ in range(args.batch_repeats):
        model(data)

for _ in tqdm(range(args.epoch_repeats), desc="Bench"):
    for _ in range(args.batch_repeats):
        model(data)


#sampler = tgloader.DynamicBatchSampler(dataset, max_num=1024, mode="node")
#loader = tgloader.DataLoader(dataset, batch_sampler = sampler)
#loader = PrefetchLoader(loader, device)
