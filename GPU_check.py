import torch

def check_available_gpus():
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("No GPUs available.")
    else:
        print(f"Number of available GPUs: {num_gpus}")
        for i in range(num_gpus):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

if __name__ == "__main__":
    check_available_gpus()