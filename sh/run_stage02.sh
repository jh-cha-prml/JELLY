# pretrain stage 02
torchrun --nproc_per_node=4 ../train.py --cfg-path ../configs/config_stage02_1.yaml

# stage 02
torchrun --nproc_per_node=4 ../train.py --cfg-path ../configs/config_stage02_2.yaml


