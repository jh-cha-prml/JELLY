# Copyright (2024) Tsinghua University, Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import soundfile as sf
import numpy as np
from transformers import WhisperFeatureExtractor
import torch.nn.functional as F
from audio import pad_or_trim, log_mel_spectrogram


class JELLYDataset(Dataset):
    def __init__(self, ann_path, cfg):
        super().__init__()
        self.model_config = cfg.config.model
        self.data_config = cfg.config.datasets
        
        self.annotation = json.load(open(ann_path, "r"))["annotation"]

        self.wav_processor = WhisperFeatureExtractor.from_pretrained(self.data_config.whisper_path)
        
        self.use_whisper = self.model_config.use_whisper
        
        self.use_emotion_label = self.model_config.use_emotion_label
        
        self.use_tltr = self.model_config.use_tltr

    def __len__(self):
        return len(self.annotation)

    def collater(self, samples):
        if self.use_whisper :
            samples_spectrogram = [s["spectrogram"] for s in samples]
            cat_spectrogram = torch.stack(samples_spectrogram, dim=0)
            samples_spectrogram_history = [s["spectrogram_history"] for s in samples if s["spectrogram_history"] is not None]
        else :
            cat_spectrogram =  None
            samples_spectrogram_history = None
        
        raw_wav = None
        padding_mask = None

        text = [s["text"] for s in samples]
        task = [s["task"] for s in samples]
        transcript = [s["transcript"] for s in samples]
        emotion = [s["emotion"] for s in samples]
        speaker = [s["speaker"] for s in samples]

        id = [s["id"] for s in samples]

        # Compute padding mask history
        audio_history = [s["audio_history"] for s in samples if s["audio_history"] is not None]
        padding_mask_history = []
        
        if audio_history: 
            for dialogue_idx in range(len(audio_history)):
                audio_history[dialogue_idx] = [torch.from_numpy(a) for a in audio_history[dialogue_idx]]
            padding_mask_history = None

        return {
            "spectrogram": cat_spectrogram, #tensor => B, ...
            "raw_wav": raw_wav,
            "padding_mask": padding_mask,
            "text": text,
            "task": task,
            "id": id,
            "spectrogram_history": samples_spectrogram_history, #list
            "audio_history": audio_history, #list
            "padding_mask_history": padding_mask_history, #list
            "transcript": transcript,
            "emotion": emotion,
            "speaker": speaker,
        }
   
    def __getitem__(self, index):
        ann = self.annotation[index]

        audio, sr = sf.read(ann["path"])
        if len(audio.shape) == 2: # stereo to mono
            audio = audio[:, 0]

        if len(audio) < sr: # pad audio to at least 1s
            sil = np.zeros(sr - len(audio), dtype=float)
            audio = np.concatenate((audio, sil), axis=0)
        audio = audio[: sr * 30] # truncate audio to at most 30s
        
        spectrogram = None
        if self.use_whisper :
            if self.use_tltr :
                spectrogram = log_mel_spectrogram(audio)
                
                single = spectrogram.ndim == 2
                spectrogram = pad_or_trim(spectrogram, 2000).to(torch.float32) # change this if the desired length is not 20s #원래는 10초임
                if not single:
                    spectrogram = spectrogram.squeeze()
                    
            else :
                spectrogram = self.wav_processor(audio, sampling_rate=sr, return_tensors="pt")["input_features"].squeeze()
                
        text = ann["text"]
        task = ann.get("task", "asr")
        
        spectrogram_history = None
        audio_history = None
        transcript = None
        emotion = None
        speaker = None
        
        if task == 'emotion_prediction_in_conversation' or task == 'only_emotion_prediction_in_conversation':
            spectrogram_history = []
            audio_history = []
            
            transcript = ann["transcript"]
            emotion = ann["emotion"]
            speaker = ann["speaker"]
            
            current_audio_path = ann["path"]
            current_audio = current_audio_path.split('/')[-1].replace('.wav','') #9_0_d2016
            
            parts = current_audio.split('_')
            current_utt_id = parts[0] #9
            current_speaker_id = parts[1] #0
            current_dialogue_id = parts[2] #d2016
            
            speaker_history = speaker.split('|')
            
            step = 0            
            if int(current_utt_id) > 9 : 
                step = int(current_utt_id) - 9
            
            for i, spk in enumerate(speaker_history) :
                target_audio = f'{step + i}_{spk}_{current_dialogue_id}.wav'
                parent = os.path.dirname(current_audio_path)
                target_path = os.path.join(parent, target_audio)
                                
                target_audio, target_sr  = sf.read(target_path)
                if len(target_audio.shape) == 2: # stereo to mono
                    target_audio = target_audio[:, 0]
                if len(target_audio) < target_sr: # pad audio to at least 1s
                    sil = np.zeros(target_sr - len(target_audio), dtype=float)
                    target_audio = np.concatenate((target_audio, sil), axis=0)
                target_audio = target_audio[: target_sr * 30] # truncate audio to at most 30s
                
                if self.use_whisper :
                    if self.use_tltr :
                        target_spectrogram = log_mel_spectrogram(target_audio)
                        
                        single = target_spectrogram.ndim == 2
                        target_spectrogram = pad_or_trim(target_spectrogram, 2000).to(torch.float32) # change this if the desired length is not 20s
                        if not single:
                            target_spectrogram = target_spectrogram.squeeze()
                        spectrogram_history.append(target_spectrogram)
                    
                    else : 
                        spectrogram_history.append(self.wav_processor(target_audio, sampling_rate=target_sr, return_tensors="pt")["input_features"].squeeze())
                
        if spectrogram_history:
            spectrogram_history = torch.stack(spectrogram_history, dim=0)
            
              
        return {
            "spectrogram": spectrogram,
            "raw_wav": audio,
            "text": text,
            "task": task,
            "id": ann["path"],
            "spectrogram_history": spectrogram_history, #tensor
            "audio_history": audio_history, #list [np, np, np ...]
            "transcript": transcript,  #str
            "emotion": emotion, #str
            "speaker": speaker, #str
        }
        
        
class JELLYDatasetExtracted(Dataset):
    def __init__(self, ann_path, cfg):
        super().__init__()
        self.model_config = cfg.config.model
        self.data_config = cfg.config.datasets
        
        self.annotation = json.load(open(ann_path, "r"))["annotation"]
        self.use_whisper = self.model_config.use_whisper
        self.use_emotion_label = self.model_config.use_emotion_label
        self.use_tltr = self.model_config.use_tltr
        
        
        if self.model_config.tltr_downsampling_rate == 40 :
            self.pooling_size = "25"
        elif self.model_config.tltr_downsampling_rate == 100 :
            self.pooling_size = "10"

    def __len__(self):
        return len(self.annotation)
    
    def pad_tensor(self, tensor, pad_size):
        if tensor.dim() == 3:
            padding = pad_size - tensor.size(1)
            if padding > 0:
                return F.pad(tensor, (0, 0, 0, padding), value=0)
            else:
                return tensor
        elif tensor.dim() == 2:
            padding = pad_size - tensor.size(0)
            if padding > 0:
                return F.pad(tensor, (0, 0, 0, padding), value=0)
            else:
                return tensor
        else:
            raise ValueError("Unsupported tensor dimension: {}".format(tensor.dim()))
        
    def process_tensor(self, data, dtype=torch.float32):
        if isinstance(data, torch.Tensor):
            return data.detach()
        else:
            return torch.tensor(data, dtype=dtype)
        
    def make_pad_mask( 
        self, lengths: torch.Tensor, max_len: int = 0, left_pad=False
    ) -> torch.Tensor:
        """
        Args:
        lengths:
            A 1-D tensor containing sentence lengths.
        max_len:
            The length of masks.
        left_pad:
            A boolean indicating whether to left pad the mask.
        Returns:
        Return a 2-D bool tensor, where masked positions
        are filled with `True` and non-masked positions are
        filled with `False`.

        >>> lengths = torch.tensor([1, 3, 2, 5])
        >>> make_pad_mask(lengths)
        tensor([[False,  True,  True,  True,  True],
                [False, False, False,  True,  True],
                [False, False,  True,  True,  True],
                [False, False, False, False, False]])
        """
        assert lengths.ndim == 1, lengths.ndim
        max_len = max(max_len, lengths.max())
        n = lengths.size(0)
        seq_range = torch.arange(0, max_len, device=lengths.device)
        expaned_lengths = seq_range.unsqueeze(0).expand(n, max_len)
        mask = expaned_lengths >= lengths.unsqueeze(-1)

        if left_pad:
            mask = mask.flip(dims=[1])

        return mask

    def collater(self, samples):
        
        text = [s["text"] for s in samples]
        task = [s["task"] for s in samples]
        transcript = [s["transcript"] for s in samples]            
        speaker = [s["speaker"] for s in samples]
        id = [s["id"] for s in samples]
        
        samples_whisper_features_history = None

        cat_whisper_features = None
        emotion = None
  
        if 'emotion_prediction_in_conversation' in task or 'only_emotion_prediction_in_conversation' in task: #emotion prediction in conversation
            if self.use_emotion_label :
                emotion = [s["emotion"] for s in samples]
            else :
                if self.use_whisper : 
                    samples_whisper_features_history = [s["whisper_features_history"] for s in samples if s["whisper_features_history"] is not None]
        else : 
            if self.use_whisper : 
                samples_whisper_features = [s["whisper_features"] for s in samples if s["whisper_features"] is not None]
                
                if self.use_tltr :
                    cat_whisper_features = torch.stack(samples_whisper_features, dim=0) 
                else :
                    max_len_whisper = max(tensor.size(0) for tensor in samples_whisper_features)

                    samples_whisper_features_padded = [self.pad_tensor(embed, max_len_whisper) for embed in samples_whisper_features]

                    cat_whisper_features = torch.stack(samples_whisper_features_padded, dim=0) 
                    cat_whisper_features = cat_whisper_features.squeeze(1)

        return {
            "whisper_features": cat_whisper_features,
            "text": text,
            "task": task,
            "id": id,
            "whisper_features_history": samples_whisper_features_history, #list => B, History, ...
            "transcript": transcript,
            "emotion": emotion,
            "speaker": speaker,
        }
   
    def __getitem__(self, index):
        ann = self.annotation[index]
        
        if "text" in ann.keys() :
            text = ann["text"]
        else :
            text = None
        task = ann.get("task", "asr")
            
        whisper_features = None
        whisper_features_history = None
        transcript = None
        emotion = None
        speaker = None
        phone_ids = None
        tgt_speech_ids = None

        if task == 'emotion_prediction_in_conversation' or task == 'only_emotion_prediction_in_conversation':
            if self.use_emotion_label :
                emotion = ann["emotion"]
                
            whisper_features_history = []
            
            transcript = ann["transcript"]
            speaker = ann["speaker"]
            
            current_audio_path = ann["path"]
            current_audio = current_audio_path.split('/')[-1].replace('.wav','') #9_0_d2016
            
            parts = current_audio.split('_')
            current_utt_id = parts[0] #9
            current_speaker_id = parts[1] #0
            current_dialogue_id = parts[2] #d2016
            
            speaker_history = speaker.split('|')
            
            step = 0            
            if int(current_utt_id) > 9 :
                step = int(current_utt_id) - 9
            
            for i, spk in enumerate(speaker_history) :
                target_audio = f'{i + step}_{spk}_{current_dialogue_id}.wav'
                parent = os.path.dirname(current_audio_path)
                target_path = os.path.join(parent, target_audio)
                
                if self.use_whisper and not self.use_emotion_label:
                    if self.use_tltr :
                        path_parts = target_path.split('/')
                        path_parts[4] += '/features_tltr_1280_' + self.pooling_size
                        whisper_features_path = '/'.join(path_parts)
                    else :
                        path_parts = target_path.split('/')
                        path_parts[4] += '/features_whisper' 
                        whisper_features_path = '/'.join(path_parts)

                    whisper_features_path = whisper_features_path.replace('.wav', '.pt')
                    whisper_features = torch.load(whisper_features_path, map_location='cpu')
                    whisper_features = whisper_features.unsqueeze(0)
                    whisper_features_history.append(whisper_features)
                    
        else : 
            path_parts = ann["path"].split('/')
            path_parts[4] += '/features_whisper' 
            whisper_features_path = '/'.join(path_parts)
            
            whisper_features_path = whisper_features_path.replace('.wav', '.pt')
            
            if self.use_whisper :
                if self.use_tltr :
                    path_parts = ann["path"].split('/')
                    path_parts[4] += '/features_tltr_1280_' + self.pooling_size
                    whisper_features_path = '/'.join(path_parts)
                    whisper_features_path = whisper_features_path.replace('.wav', '.pt')
                    whisper_features = torch.load(whisper_features_path, map_location='cpu')
                else :
                    whisper_features = torch.load(whisper_features_path, map_location='cpu')
                
        return {
            "whisper_features": whisper_features,
            "text": text,
            "task": task,
            "id": ann["path"],
            "whisper_features_history": whisper_features_history, #list
            "transcript": transcript,  #str
            "emotion": emotion, #str
            "speaker": speaker, #str
        }