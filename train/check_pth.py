import torch
from torch import nn
import os

if __name__ == "__main__":

    path = "./model_checkpoints/qatresnet50_4Bit_imagenet_lr0.1_sgd_batch_256_epoch100/halut/lutmu_kn2col_custom_C_K_lr_0.0005/checkpoints"
    files = os.listdir(path)
    
    for file in files:
        if "model_best" in file or "checkpoint.pth" in file:
            file_path = os.path.join(path, file)
            state_dict = torch.load(file_path, map_location=torch.device('cpu'))
            
        
            if 'halut_modules' in state_dict:
                print(file, state_dict['epoch'], state_dict['args'].batch_size, state_dict['args'].lr, state_dict['halut_modules'].keys())
            else:
                print(file, state_dict['epoch'], state_dict['args'].batch_size, state_dict['args'].lr, "No halut modules found")


    