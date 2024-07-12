import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist
import os
import argparse

def check_available_gpus():
    try:
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            print("No GPUs available.")
        else:
            print(f"Number of available GPUs: {num_gpus}")
            for i in range(num_gpus):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

        return num_gpus
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Make sure PyTorch is installed with ROCm support.")

def init_process(rank, world_size, master_addr, master_port, train_fn, backend='nccl'):
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    train_fn(rank, world_size)

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train(rank, world_size):
    # Set device and seed
    torch.manual_seed(0)
    device = torch.device(f'cuda:{rank % torch.cuda.device_count()}')
    
    # Load data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=64, sampler=sampler)
    
    # Build model and move it to GPU
    model = SimpleNN().to(device)
    ddp_model = DDP(model, device_ids=[rank % torch.cuda.device_count()])
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001, momentum=0.9)
    
    # Train the model
    ddp_model.train()
    for epoch in range(10):
        sampler.set_epoch(epoch)
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = ddp_model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(dataloader.dataset)}] Loss: {loss.item()}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--nodes', type=int, default=1, help='number of nodes')
    parser.add_argument('--gpus', type=int, default=1, help='number of gpus per node')
    parser.add_argument('--nr', type=int, default=0, help='ranking within the nodes')
    parser.add_argument('--master_addr', type=str, default='localhost', help='address of the master node')
    parser.add_argument('--master_port', type=str, default='29500', help='port of the master node')
    args = parser.parse_args()
    
    world_size = args.gpus * args.nodes
    rank = args.nr * args.gpus
    for gpu in range(args.gpus):
        mp.spawn(init_process, nprocs=args.gpus, args=(world_size, args.master_addr, args.master_port, train))
