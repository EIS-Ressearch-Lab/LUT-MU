python training/train.py --device cuda:0 --model quant_resnet9 --cifar10 \
                         --lr 0.01 --epochs 1 --amp --batch-size 256 -j 8 \
                         --bitwidth 4 --opt adam --lr-scheduler cosineannealinglr \
                         --output-dir ./model_checkpoints/test