python retraining.py 0 -single -workdir ./model_checkpoints/output  \
                       -checkpoint ./model_checkpoints/output/checkpoint.pth \
                       -testname testname \
                       --kn2col --lutmu -batch_size 256 --epochs 25 -j 8 --lr 0.001 \
                       --kc_config ./configs/quant_resnet9_num_C_8_K_16.json