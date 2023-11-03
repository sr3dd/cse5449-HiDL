import argparse
import os
import time

import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist

# import torchvision module to handle image manipulation
import torchvision
from torchvision import models
import torchvision.transforms as transforms


def env2int(env_list, default = -1):
    for e in env_list:
        val = int(os.environ.get(e, -1))
        if val >= 0: return val
    return default
my_local_rank = env2int(['MPI_LOCALRANKID', 'OMPI_COMM_WORLD_LOCAL_RANK', 'MV2_COMM_WORLD_LOCAL_RANK'], 0)
os.environ["CUDA_VISIBLE_DEVICES"]=str(my_local_rank)


train_set = torchvision.datasets.CIFAR10(
      root = './data/cifar10',
      train = True,
      download = False,
      transform = transforms.Compose([
          transforms.ToTensor()                                 
      ])
)

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        rank:int
    ) -> None:
        self.model = model.to('cude:0')
        self.train_data = train_data
        self.optimizer = optimizer
        self.rank = rank
        self.model = DDP(self.model, device_ids = [rank])

        self.batches = []
        self.epoch_times = []
        self.throughputs = []

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = F.cross_entropy(output, targets)
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        #print(f"[Rank {self.rank}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}",flush=True)
        for source, targets in self.train_data:
            time_start = time.time()
            source = source.to(f'cuda:{self.rank}')
            targets = targets.to(f'cuda:{self.rank}')
            self.batches.append(source.size()[0])
            self._run_batch(source, targets)
            time_end = time.time() - time_start
            self.throughputs.append(source.size()[0]/time_end)

    def train(self, max_epochs: int):

        for epoch in range(max_epochs):
            time_start = time.time()
            self._run_epoch(epoch)
            self.epoch_times.append(time.time() - time_start)

    def print(self):
        print(f'{self.rank}\t{np.mean(self.batches[0])}\t{len(self.throughputs)}\t{np.mean(self.throughputs)}\t{np.mean(self.epoch_times)}', flush=True)


def load_train_objs():
    train_set = torchvision.datasets.CIFAR10(
      root = './data/cifar10',
      train = True,
      download = False,
      transform = transforms.Compose([
          transforms.ToTensor()                                 
      ])
    )

    model = models.resnet34()  # load your model
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    return train_set, model, optimizer


def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        sampler=DistributedSampler(dataset),
        shuffle=False
    )


def main(total_epochs, batch_size, rank):
    dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = Trainer(model, train_data, optimizer, rank)
    trainer.train(total_epochs)
    trainer.print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pytorch Distributed Training.')

    parser.add_argument(
        '--batch-size', action='store', default=1024, type=int,
        help='Batch size (default: 32)')

    parser.add_argument(
        '--num-epoch', action='store', default=1, type=int,
        help='Number of epochs (default: 1)')

    parser.add_argument(
        '--num-iters', action='store', default=100, type=int,
        help='Number of iterations (default: 100)')

    parser.add_argument('--strong-scale', action='store_true', default=False,
                        help='Perform strong scaling if set, otherwise weak scaling.')

    args = parser.parse_args()
    dist.init_process_group(backend='mpi')
    size = dist.get_world_size()
    rank = dist.get_rank()

    bsize = args.batch_size if not args.strong_scale else int(args.batch_size/size)
    main(args.num_epoch, bsize, rank)