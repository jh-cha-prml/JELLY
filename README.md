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

## Training & Inference
```
### stage 01 Emotion-Text Alignment
cd JELLY_llm
bash ./sh/run_stage01.sh

### stage 02 Emotional Context Reasoning
cd JELLY_llm
bash ./sh/run_stage02.sh

### stage 03 Emotional Context-Aware Speech Synthesis
cd JELLY_tts
bash ./sh/run_stage03.sh

```

## To-do list
- [x] add requirements.txt
- [x] add stage01, stage02 dataset json
- [x] add stage01, stage02 training code
- [x] add stage03 dataset txt
- [x] add stage03 training code
- [x] add inference code
- [ ] whisper feature extraction code
- [ ] more README update

## References
This implementation was developed based on the following repository:
* SALMONN: <https://github.com/bytedance/SALMONN> (for stage 01-02 architecture backbone)
* Whisper-AT: <https://github.com/YuanGongND/whisper-at> (for TLTR module)
* BLSP-Emo: <https://github.com/cwang621/blsp-emo> (for partial LoRA module)
* DailyTalk: <https://github.com/keonlee9420/DailyTalk> (for stage 03 architecture backbone)
