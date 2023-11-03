import argparse
import time

import numpy as np
import pandas as pd

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
from torchvision import models
import torchvision.transforms as transforms

torch.cuda.manual_seed(0)
torch.manual_seed(0)

train_set = torchvision.datasets.CIFAR10(
      root = './data/cifar10',
      train = True,
      download = True,
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

def get_accuracy(model,dataloader):
    count=0
    correct=0

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            images = batch[0].cuda()
            labels = batch[1].cuda()
            preds=model(images)
            batch_correct=preds.argmax(dim=1).eq(labels).sum().item()
            batch_count=len(batch[0])
            count+=batch_count
            correct+=batch_correct
    
    model.train()
    return correct/count

def main(batch_size, warmup_batches, num_iters):

    print(f'batch size: {batch_size}')
    df = []

    model = models.resnet34()
    loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    lr = 0.001
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.cuda()

    model.train()

    # warmup
    for _ in range(warmup_batches):
        images, labels = next(iter(loader))
        images, labels = images.cuda(), labels.cuda()
        optimizer.zero_grad()
        preds = model(images)
        loss = F.cross_entropy(preds, labels)
        loss.backward()
        optimizer.step()

    throughputs = []
    df = []

    # start of iterations
    for iteration in range(num_iters):
        
        images, labels = next(iter(loader))
        
        time_start = time.time()

        images, labels = images.cuda(), labels.cuda()
        optimizer.zero_grad()
        preds = model(images)
        loss = F.cross_entropy(preds, labels)
        loss.backward()
        optimizer.step()
        
        time_iter = time.time() - time_start
        
        throughput = images.size()[0]/time_iter
        throughputs.append(throughput)
        df.append({'bsize': batch_size, 'iteration': iteration, 'throughput': throughput})
    
    avg_throughput = np.mean(throughputs)
    print(f'Iteration {iteration}: Avg. Throughput = {avg_throughput} img/s')

    test_loader = torch.utils.data.DataLoader(test_set, batch_size = 64)
    print(f'Test set accuracy {get_accuracy(model, test_loader)}', flush=True)

    return df

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Single GPU ResNet-34 Training Benchmark',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--num-warmup-batches', type=int, default=10,
                        help='number of warm-up batches (do not count towards throughput measurement)')
    
    parser.add_argument('--num-iters', type=int, default=100,
                        help='number of benchmark iterations')

    args = parser.parse_args()

    if not torch.cuda.is_available():
        print('GPU device not present.')
    else:
        batch_sizes = [pow(2,x) for x in range(14)][1:]
        
        all_batch_rows = []

        for batch_size in batch_sizes:
            batch_rows = main(batch_size, args.num_warmup_batches, args.num_iters)
            all_batch_rows += batch_rows

        df = pd.DataFrame.from_records(all_batch_rows)
        df.to_csv('./output/single_gpu.csv', sep='\t')
