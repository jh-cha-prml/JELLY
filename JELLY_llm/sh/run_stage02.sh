# # # # # # # # # # # # # # # # # # # # pretraining STAGE 02 # # # # # # # # # # # # # # # # # # # #
# torchrun --nproc_per_node=4 ./train.py --cfg-path ./configs/config_stage02_1.yaml

# # # # # # # # # # # # # # # # # # # # STAGE 02 # # # # # # # # # # # # # # # # # # # #
# torchrun --nproc_per_node=4 ./train.py --cfg-path ./configs/config_stage02_2.yaml

# # # # # # # # # # # # # # # # # # # # STAGE 02 inference # # # # # # # # # # # # # # # # # # # #
torchrun --nproc_per_node=4 ./train.py --cfg-path ./configs/test_config_stage02.yaml
