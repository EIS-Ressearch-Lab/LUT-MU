import torch
from torch import nn
import os

if __name__ == "__main__":

    # path = "model_checkpoints/quant_resnet9_4Bit_cifar10_lr0.01_batch_256_epoch1/halut/lutmu_kn2col_num_C_8_K_16_lr_0.001/checkpoints"
    path = "model_checkpoints/output/halut/kn2col/checkpoints"
    # path = "model_checkpoints/output/halut/im2col/checkpoints"
    files = os.listdir(path)
    
    for file in files:
        if "model_best" in file or "checkpoint.pth" in file:
            file_path = os.path.join(path, file)
            state_dict = torch.load(file_path, map_location=torch.device('cpu'))
            
        
            if 'halut_modules' in state_dict:
                print(file, state_dict['epoch'], state_dict['args'].batch_size, state_dict['args'].lr, state_dict['halut_modules'].keys())
                # print(state_dict['model'].keys())
                for module_name in state_dict['halut_modules'].keys():
                    print(f"Module: {module_name}")
                    dims = state_dict['model'][f"{module_name}.dims"]
                    print(f"  Dimensions: {dims}")


    