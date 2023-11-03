import argparse
import os
import time

import numpy as np

# import standard PyTorch modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# import torchvision module to handle image manipulation
import torchvision
from torchvision import models
import torchvision.transforms as transforms


#Modules for Distributed Training
import horovod.torch as hvd
from filelock import FileLock

parser = argparse.ArgumentParser(description='Distibuted Training Using Horovod.')

parser.add_argument(
    '--batch-size', action='store', default=32, type=int,
    help='Batch size (default: 32)')

parser.add_argument(
    '--num-iters', action='store', default=100, type=int,
    help='Number of iterations (default: 100)')

parser.add_argument('--strong-scale', action='store_true', default=False,
                    help='Perform strong scaling if set, otherwise weak scaling.')

args = parser.parse_args()

"""The following library call downloads the training set and puts it into data/FashionMNIST, and prepares the dataset to be passed into a pytorch as a tensor."""

# Use standard FashionMNIST dataset
torch.cuda.manual_seed(0)
torch.manual_seed(0)
hvd.init()

bsize = args.batch_size if not args.strong_scale else int(args.batch_size/hvd.size())

torch.cuda.set_device(hvd.local_rank())

with FileLock(os.path.expanduser("~/.horovod_lock")):
  train_set = torchvision.datasets.CIFAR10(
      root = './data/cifar10',
      train = True,
      download = False,
      transform = transforms.Compose([
          transforms.ToTensor()                                 
      ])
  )

  test_set = torchvision.datasets.CIFAR10(
      root = './data/cifar10',
      train = False,
      download = False,
      transform = transforms.Compose([
          transforms.ToTensor()                                 
      ])
  )


#Sample the batch ids based on the rank
train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_set, num_replicas=hvd.size(), rank=hvd.rank())
test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_set, num_replicas=hvd.size(), rank=hvd.rank())

def get_accuracy(model,dataloader):
    count=0
    correct=0
    model.eval()
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch[0]
            labels = batch[1]
            preds=network(images)
            batch_correct=preds.argmax(dim=1).eq(labels).sum().item()
            batch_count=len(batch[0])
            count+=batch_count
            correct+=batch_correct

    model.train()
    return correct/count


lr=0.001
shuffle=True
epochs=1

network = models.resnet34()
#network.cuda()

loader = torch.utils.data.DataLoader(train_set, batch_size = bsize, sampler=train_sampler)
optimizer = optim.Adam(network.parameters(), lr=lr)


# Horovod: broadcast parameters & optimizer state.
hvd.broadcast_parameters(network.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(optimizer, root_rank=0)

# Horovod: wrap optimizer with DistributedOptimizer.
optimizer = hvd.DistributedOptimizer(optimizer,
                                     named_parameters=network.named_parameters(),
                                     op=hvd.Average)

## To average performance metric
def metric_average(val, name):
    tensor = torch.tensor(val)
    avg_tensor = hvd.allreduce(tensor, name=name)
    return avg_tensor.item()

# set the network to training mode
network.train()

#iter_num = 0

#print(f'Rank {hvd.rank()}, loader: {len(loader)}')

throughputs = []
epoch_times = []
train_accuracies = []
batches = []

for epoch in range(epochs):
    time_start = time.time()

    for i, batch in enumerate(loader):
        # print(f'Rank {hvd.rank()}, batch num: {i}')
        # print(f'Rank {hvd.rank()}, batch size: {batch[0].size()[0]}')
        time_start_iter = time.time()
        images = batch[0]
        labels = batch[1]
        preds = network(images)
        loss = F.cross_entropy(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        time_end_iter = time.time() - time_start_iter

        throughput = batch[0].size()[0]/time_end_iter
        throughputs.append(throughput)
        batches.append(batch[0].size()[0])
    
    time_end = time.time() - time_start
    epoch_times.append(time_end)
    train_accuracies.append(get_accuracy(network, loader))

print(f'{hvd.rank()}\t{np.mean(batches)}\t{len(throughputs)}\t{np.mean(throughputs)}\t{np.mean(epoch_times)}', flush=True)
