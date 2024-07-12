import os
import torch
from torch import nn
import torch.utils.data as data
from torchvision import datasets, transforms
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import logging
import argparse

# Configure logging
def log(args):
    logging.basicConfig(
        filename=args.logname,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

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

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device_id):
    model.train()
    total_step = len(train_loader)
    best_val_acc = 0.0
    logging.info("Training the SimpleNN model for %d epochs...", num_epochs)

    for epoch in range(num_epochs):
        logging.info("Epoch %d/%d", epoch + 1, num_epochs)
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device_id)
            labels = labels.to(device_id)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99 and device_id == 0:
                logging.info('[%d, %5d] loss: %.3f', epoch + 1, i + 1, running_loss / 100)
                running_loss = 0.0

        val_acc = evaluate_model(model, val_loader, device_id)
        logging.info("Epoch: %d, Validation Accuracy: %.4f", epoch + 1, val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_simple_nn_model.pth")

        logging.info('Finished Training Step %d', epoch + 1)

    logging.info('Finished Training. Best Validation Accuracy: %.4f', best_val_acc)

def evaluate_model(model, val_loader, device_id):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device_id)
            labels = labels.to(device_id)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

def simple_nn_train(args):
    local_rank = int(os.environ['SLURM_LOCALID'])
    os.environ['MASTER_ADDR'] = str(os.environ['HOSTNAME'])
    os.environ['MASTER_PORT'] = "29500"
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
        world_size=args.world_size,
        rank=int(os.environ['RANK'])
    )
    print("SLURM_LOCALID/local_rank:{}, dist_rank:{}".format(local_rank, dist.get_rank()))

    print(f"Start running basic DDP example on rank {local_rank}.")
    device_id = local_rank % torch.cuda.device_count()
    
    # Create DataLoader for training and validation
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    val_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

    train_sampler = data.DistributedSampler(train_dataset, num_replicas=args.world_size, rank=local_rank)
    val_sampler = data.DistributedSampler(val_dataset, num_replicas=args.world_size, rank=local_rank)

    train_loader = data.DataLoader(dataset=train_dataset, batch_size=64, sampler=train_sampler)
    val_loader = data.DataLoader(dataset=val_dataset, batch_size=64, sampler=val_sampler)

    # Create SimpleNN model
    model = SimpleNN()
    model.to(device_id)
    model = DDP(model, device_ids=[device_id])

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, args.num_epochs, device_id=device_id)
    dist.destroy_process_group()

def simple_nn_ddp(args):
    log(args=args)
    args.world_size = int(os.environ['SLURM_NTASKS'])
    # mp.spawn(simple_nn_train, nprocs=args.gpus, args=(args,))
    simple_nn_train(args=args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--nodes', type=int, default=1, help='number of nodes')
    parser.add_argument('--gpus', type=int, default=1, help='number of gpus per node')
    parser.add_argument('--nr', type=int, default=0, help='ranking within the nodes')
    parser.add_argument('--logname', type=str, default='training.log', help='log file name')
    parser.add_argument('--num_epochs', type=int, default=10, help='number of epochs to train')
    parser.add_argument('--pretrained', type=bool, default=False, help='use pretrained model or not')
    args = parser.parse_args()

    args.world_size = args.gpus * args.nodes
    mp.spawn(simple_nn_ddp, nprocs=args.gpus, args=(args,))
