# # # # # # # # # # # # # # # # # # # # STAGE 01 - 1 # # # # # # # # # # # # # # # # # # # #
# torchrun --nproc_per_node=4 ./train.py --cfg-path ./configs/config_stage01_1.yaml

# # # # # # # # # # # # # # # # # # # # STAGE 01 - 2 # # # # # # # # # # # # # # # # # # # #
torchrun --nproc_per_node=4 ./train.py --cfg-path ./configs/config_stage01_2.yaml

