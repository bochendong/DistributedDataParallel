import os
import torch
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

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


def train_model(ddp_model, optimizer, criterion, dataloader, device_id):
    ddp_model.train()
    for epoch in range(10):
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device_id), target.to(device_id)
            
            optimizer.zero_grad()
            output = ddp_model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(dataloader.dataset)}] Loss: {loss.item()}')

def train(args):
    local_rank = int(os.environ['SLURM_LOCALID'])
    os.environ['MASTER_ADDR'] = os.environ['MASTER_ADDR']
    os.environ['MASTER_PORT'] = os.environ['MASTER_PORT']
    os.environ['WORLD_SIZE'] = os.environ['SLURM_NTASKS']
    os.environ['RANK'] = os.environ['SLURM_PROCID']
    
    print("MASTER_ADDR:{}, MASTER_PORT:{}, WORLD_SIZE:{}, WORLD_RANK:{}, local_rank:{}".format(os.environ['MASTER_ADDR'], 
                                                    os.environ['MASTER_PORT'], 
                                                    os.environ['WORLD_SIZE'], 
                                                    os.environ['RANK'],
                                                    local_rank))
    dist.init_process_group(                                   
    	backend='nccl',                                         
   		init_method='env://',                                   
    	world_size=int(args.world_size),                              
    	rank=int(os.environ['RANK'])                                               
    )
    print("SLURM_LOCALID/local_rank:{}, dist_rank:{}".format(local_rank, dist.get_rank()))

    print(f"Start running basic DDP example on rank {local_rank}.")

    device_id = local_rank % torch.cuda.device_count()
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=64, sampler=sampler)

    model = SimpleNN()
    model.to(device_id)
    model = DDP(model, device_ids=[device_id])

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    train_model(model, optimizer, criterion, dataloader, device_id)
    dist.destroy_process_group()

def TrainSimpleNNDDP(args):
    args.world_size = int(os.environ['SLURM_NTASKS'])
    train(args=args)

if __name__ == '__main__':
    import argparse

    def parse_args():
        parser = argparse.ArgumentParser(description='PyTorch Simple NN DDP Example')
        parser.add_argument('--gpus', type=int, default=8, help='Number of gpus per device')
        parser.add_argument('--nodes', type=int, default=1, help='Number of device')
        args = parser.parse_args()
        return args

    args = parse_args()
    TrainSimpleNNDDP(args)
