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
import torch_geometric.profile as tgprof
import aottools

parser = argparse.ArgumentParser()
parser.add_argument("--aot", action="store_true")
parser.add_argument("--prefetch", action="store_true")
parser.add_argument("--loader_type", type=str, default="gnm")
parser.add_argument("--work_type", type=str, default="test")
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--dim_size", type=int, default=512)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--warm_repeats", type=int, default=10)
parser.add_argument("--bench_repeats", type=int, default=50)
args = parser.parse_args()
print("------------------------------")
print(args)
print("------------------------------")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = tgsets.ZINC(".dataset/ZINC", True, split="train")

class ModelGraph(nn.Module):
    def __init__(self, dim: int, num_layer: int):
        super().__init__()
        self.embed = nn.Embedding(22, dim)
        self.gnn = tgnn.GIN(dim, dim, 5, dim, norm="layernorm")
        self.agg = tgnn.DeepSetsAggregation(nn.Identity(), nn.Linear(dim, 1))

    def forward(self, batch):
        h = self.embed(batch.x.flatten())
        h = self.gnn(h, batch.edge_index)
        return self.agg(h, batch.batch)

model = ModelGraph(args.dim_size, 3).to(device)
if args.aot: 
    model = tg.compile(model, mode="max-autotune")

def test():
    model.eval()
    for data in loader:
        if args.prefetch: d = data
        else: d = data.to(device)
        model(d)

loss = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
def train():
    model.train()
    for data in loader:
        if args.prefetch: d = data
        else: d = data.to(device)
        optimizer.zero_grad()
        loss(model(d), d.y).backward()
        optimizer.step()
    
if args.work_type == "train":
    work = train
else:
    work = test

if args.loader_type == "constant":
    loader0 = tgloader.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    data = next(iter(loader0))
    loader = tgloader.DataLoader([data] * (len(dataset) // args.batch_size), batch_size=1, shuffle=False, num_workers=args.num_workers)
elif args.loader_type == "normal":
    loader = tgloader.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
elif args.loader_type == "gnm":
    loader = aottools.GnmLoader(dataset, args.batch_size, shuffle=True, num_workers=args.num_workers)
else:
    raise NotImplementedError

if args.prefetch:
    loader = aottools.PrefetchLoader(loader, device)

print("Warmup")
for _ in range(args.warm_repeats): 
    work()

print("Loader ", end="", flush=True)
with tgprof.timeit(True, args.bench_repeats):
    for _ in range(args.bench_repeats): 
        for data in loader:
            pass

print("Model  ", end="", flush=True)
datalist = list(loader)
with tgprof.timeit(True, args.bench_repeats):
    for _ in range(args.bench_repeats): 
        for data in datalist:
            model(data.to(device))

print("Total  ", end="", flush=True)
with tgprof.timeit(True, args.bench_repeats):
    for _ in range(args.bench_repeats): 
        for data in loader:
            model(data.to(device))