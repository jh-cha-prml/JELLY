# # # # # # # # # # # # # # # # # # # # STAGE 03 # # # # # # # # # # # # # # # # # # # #

# #preprocess
# python3 prepare_align.py --dataset DailyTalk
# python3 preprocess.py --dataset DailyTalk 

# # train
# CUDA_VISIBLE_DEVICES=0 python3 train.py --dataset DailyTalk --use_amp 

# # inference
# RESTORE_STEP=275000
# CUDA_VISIBLE_DEVICES=0 python3 synthesize.py --source preprocessed_data/DailyTalk/test_frame.txt --restore_step ${RESTORE_STEP} --mode batch --dataset DailyTalk

# # inference (based on stage 02 prediction )
RESTORE_STEP=275000
CUDA_VISIBLE_DEVICES=0 python3 synthesize.py --source preprocessed_data/DailyTalk/test_frame_pred.txt --restore_step ${RESTORE_STEP} --mode batch --dataset DailyTalk

