# JELLY: Joint Emotion Recognition and Context Reasoning with LLMs for Conversational Speech Synthesis

## [Demo Page](https://jh-cha-prml.github.io/JELLY/) 
**Jun-Hyeok Cha, Seung-Bin Kim, Hyung-Seok Oh, and Seong-Whan Lee**

## Clone our repository
```
git clone https://github.com/jh-cha-prml/JELLY
cd ./JELLY
```

## Install the requirements
```
pip install -r requirements.txt
```

## Training
```
bash ./sh/run_stage01.sh
bash ./sh/run_stage02.sh

```

## To-do list
- [x] add requirements.txt
- [x] add stage01, stage02, stage03 dataset json
- [x] add stage01, stage02 training code
- [ ] add stage03 training code
- [ ] add inference code

## References
This implementation was developed based on the following repository:
* SALMONN: <https://github.com/bytedance/SALMONN> (for architecture backbone)
* Whisper-AT: <https://github.com/YuanGongND/whisper-at> (for TLTR module)
* BLSP-Emo: <https://github.com/cwang621/blsp-emo> (for partial LoRA module)
