import os
import argparse

from MultiDeviceMultiGPU import TrainSimpleNNDDP

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Simple NN DDP Example')
    parser.add_argument('--gpus', type=int, default=8, help='Number of gpus per device')
    parser.add_argument('--nodes', type=int, default=1, help='Number of device')
    args = parser.parse_args()
    return args

def main(args):
    args.output = os.path.join(args.output, args.task)
    os.makedirs(args.output, exist_ok=True)
    
    if (args.nodes >= 2):
        TrainSimpleNNDDP()
    else:
        raise "No such task."
    
if __name__ == '__main__':
    args = parse_args()
    main(args=args)