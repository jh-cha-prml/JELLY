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

import logging
import json
import contextlib
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaTokenizer, StoppingCriteriaList
# from peft import LoraConfig, TaskType, get_peft_model # no plora
from peft import TaskType # plora
from .plora import LoraConfig, LoraModel # plora


from dist_utils import get_rank

from .Qformer import BertConfig, BertLMHeadModel
from .modeling_llama import LlamaForCausalLM
from .modeling_whisper import WhisperModel
from .utils import StoppingCriteriaSub, concat_all_gather, all_gather_with_grad
from funasr import AutoModel
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer
from .tltr import ATModel
import skimage.measure
import numpy as np




class JELLY(nn.Module):
    @classmethod
    def init_bert_tokenizer(cls, truncation_side="right"):
        bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", truncation_side=truncation_side)
        # self.tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        return bert_tokenizer
            # Example::
            # >>> from transformers import BertTokenizer, BertLMHeadModel, BertConfig
            # >>> import torch
            # >>> tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
            # >>> config = BertConfig.from_pretrained("bert-base-cased")
            # >>> model = BertLMHeadModel.from_pretrained('bert-base-cased', config=config)
            # >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
            # >>> outputs = model(**inputs)
            # >>> prediction_logits = outputs.logits
            
    @classmethod
    def init_speech_Qformer(cls, num_query_token, speech_width, num_hidden_layers=2):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.num_hidden_layers = num_hidden_layers 
        encoder_config.encoder_width = speech_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 1
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens

    @property
    def device(self):
        return list(self.parameters())[0].device
    
    
    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    def __init__(
        self,
        use_extracted_feature=False,
        use_whisper=True,
        use_tltr=False,
        whisper_config_d_model=1280,
        tltr_embed_dim=1280,
        tltr_downsampling_rate=40,
        llama_path="",
        whisper_path="",
        freeze_whisper=True,
        freeze_tltr=False,
        freeze_lora=False,
        freeze_text_plora=False,
        freeze_emotion_plora=False,

        use_speech_Qformer=True,
        num_speech_query_token=25,
        freeze_speech_QFormer=False,
        
        speech_llama_proj_model="",
        freeze_speech_llama_proj=False,

        lora=True,
        lora_rank=8,
        lora_alpha=32,
        lora_dropout=0.1,
        
        use_plora=False,
        
        use_text_plora=False,
        text_plora_rank=8,
        text_plora_alpha=32, 
        text_plora_dropout=0.1,
        text_plora_target_modules="",
        
        use_emotion_plora=False,
        emotion_plora_rank=8,
        emotion_plora_alpha=32,
        emotion_plora_dropout=0.1,
        emotion_plora_target_modules="",
        
        target_speech_vocab_size=1024,
        use_input_embeds=False,
        emb_dim=256,
        multi_prompt=False,
        dialogue_prompt=False,
        use_emotion_label=False,
        prompt_path="",
        prompt_template="",
        max_txt_len=128,
        end_sym="</s>",
        low_resource=False,  # use 8 bit
        device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
    ):
        super().__init__()
        self.use_extracted_feature = use_extracted_feature
        self.use_whisper = use_whisper
        self.use_tltr = use_tltr
        self.use_speech_Qformer = use_speech_Qformer
        self.lora = lora
        
        self.use_plora=use_plora
        self.use_text_plora=use_text_plora
        self.text_plora_rank=text_plora_rank
        self.text_plora_alpha= text_plora_alpha
        self.text_plora_dropout=text_plora_dropout
        self.text_plora_target_modules=text_plora_target_modules
        self.use_emotion_plora=use_emotion_plora
        self.emotion_plora_rank=emotion_plora_rank
        self.emotion_plora_alpha=emotion_plora_alpha
        self.emotion_plora_dropout=emotion_plora_dropout
        self.emotion_plora_target_modules=emotion_plora_target_modules
        
        self.target_speech_vocab_size = target_speech_vocab_size
        self.use_input_embeds = use_input_embeds
        self.emb_dim = emb_dim
        self.multi_prompt = multi_prompt
        self.dialogue_prompt = dialogue_prompt
        self.use_emotion_label = use_emotion_label
        self.max_txt_len = max_txt_len
        self.end_sym = end_sym
        self.low_resource = low_resource
        
        self.whisper_config_d_model = whisper_config_d_model 
        self.tltr_embed_dim = tltr_embed_dim
        self.tltr_downsampling_rate = tltr_downsampling_rate
        
        logging.info('Loading LLaMA Tokenizer')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_path, use_fast=False)
        
        self.llama_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.llama_tokenizer.padding_side = "right"

        logging.info('Loading LLaMA Model')
        if self.low_resource:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_path,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map={"": device_8bit},
            )
        else:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_path,
                torch_dtype=torch.float16,
            )

        self.llama_model.resize_token_embeddings(len(self.llama_tokenizer))
                    
        logging.info('Loading LLaMA Done')
        
        if self.use_input_embeds: 
            self.emb_linear = nn.Linear(self.emb_dim, self.llama_model.config.hidden_size)
            self.emb_linear.weight.data.normal_(mean=0.0, std=0.01)
            self.emb_linear.bias.data.zero_()

        if self.lora:
            if self.use_plora :
                
                plora_config_list = []
                plora_adapter_list = []
                
                if self.use_text_plora :
                    self.text_plora_config = LoraConfig(
                        r=self.text_plora_rank,
                        lora_alpha=self.text_plora_alpha,
                        target_modules=self.text_plora_target_modules.split(","),
                        lora_dropout=self.text_plora_dropout,
                        bias="none"
                    )
                    plora_config_list.append(self.text_plora_config)
                    plora_adapter_list.append("text")
                    
                if self.use_emotion_plora :
                    self.emotion_plora_config = LoraConfig(
                        r=self.emotion_plora_rank,
                        lora_alpha=self.emotion_plora_alpha,
                        target_modules=self.emotion_plora_target_modules.split(","),
                        lora_dropout=self.emotion_plora_dropout,
                        bias="none"
                    )
                    plora_config_list.append(self.emotion_plora_config)
                    plora_adapter_list.append("emotion")
                    
                    
                if len(plora_config_list) == 0 : 
                    raise ValueError("Error: plora_scope_list is empty.")
                else :
                    self.llama_model = LoraModel(self.llama_model, plora_config_list, plora_adapter_list)
                    print(f'Partial LoRA 적용 => {plora_adapter_list}')
                    
            else :
                self.peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM, 
                    inference_mode=False, 
                    r=lora_rank, 
                    lora_alpha=lora_alpha, 
                    lora_dropout=lora_dropout,
                )
                self.llama_model = get_peft_model(self.llama_model, self.peft_config)
            
            if self.use_plora :
                if freeze_text_plora :
                    for name, param in self.llama_model.named_parameters():
                        if 'lora_A.text' in name or 'lora_B.text' in name :
                            param.requires_grad = False
                            # logging.info(f'Freezing {name}')
                            print(f'Freezing {name}')
                
                if freeze_emotion_plora :
                    for name, param in self.llama_model.named_parameters():
                        if 'lora_A.emotion' in name or 'lora_B.emotion' in name :
                            param.requires_grad = False
                            # logging.info(f'Freezing {name}')
                            print(f'Freezing {name}')
                            
                if freeze_text_plora and freeze_emotion_plora : 
                    for name, param in self.llama_model.named_parameters():
                        if 'rotary_emb.inv_freq' in name :
                            param.requires_grad = False
                            # logging.info(f'Freezing {name}')
                            print(f'Freezing {name}')
                            
            else :
                if freeze_lora :
                    # Freeze all parameters
                    for param in self.llama_model.parameters():
                        param.requires_grad = False
                    logging.info('LoRA Freezing')
                else :
                    logging.info('LoRA Training')
                # self.llama_model.print_trainable_parameters()
                
            
        if self.use_whisper :
            if self.use_extracted_feature :
                self.ln_speech = nn.LayerNorm(self.whisper_config_d_model)
                logging.info("Not loading Whisper Model, just using the extracted features by Whisper")
            else :
                assert whisper_path
                logging.info('')
                self.speech_encoder = WhisperModel.from_pretrained(whisper_path).encoder
                self.ln_speech = nn.LayerNorm(self.speech_encoder.config.d_model)
                if freeze_whisper:
                    for name, param in self.speech_encoder.named_parameters():
                        param.requires_grad = False
                    self.speech_encoder.eval()
                    logging.info("freeze Whisper")
                    
        if self.use_tltr :
            self.tltr_encoder = ATModel(rep_dim=self.tltr_embed_dim)
            
            if freeze_tltr :
                for name, param in self.tltr_encoder.named_parameters():
                    param.requires_grad = False
                print("freeze tltr module!")
                     
        if self.use_speech_Qformer:
            if self.use_extracted_feature:
                self.speech_Qformer, self.speech_query_tokens = self.init_speech_Qformer(
                    num_query_token=num_speech_query_token, speech_width=self.whisper_config_d_model
                )
                    
            else :
                self.speech_Qformer, self.speech_query_tokens = self.init_speech_Qformer(
                    num_query_token=num_speech_query_token, speech_width=self.speech_encoder.config.d_model
                )
                    
            self.speech_Qformer.bert.embeddings.word_embeddings = None
            self.speech_Qformer.bert.embeddings.position_embeddings = None
            for layer in self.speech_Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
                
            self.speech_Qformer.cls = None
                
            if freeze_speech_QFormer:
                for name, param in self.speech_Qformer.named_parameters():
                    param.requires_grad = False
                self.speech_Qformer.eval()
                self.speech_query_tokens.requires_grad = False
                logging.info("freeze Speech QFormer")
                print("freeze QFormer module!")

            logging.info('Loading speech LLAMA proj')
            if self.use_speech_Qformer :
                self.speech_llama_proj = nn.Linear(
                    self.speech_Qformer.config.hidden_size, self.llama_model.config.hidden_size
                )
                            
            if speech_llama_proj_model:
                logging.info("Loading speech LLAMA proj from {}".format(speech_llama_proj_model))
                speech_llama_proj_weight = torch.load(speech_llama_proj_model, map_location="cpu")
                self.load_state_dict(speech_llama_proj_weight['model'], strict=False)
            if freeze_speech_llama_proj:
                for name, param in self.speech_llama_proj.named_parameters():
                    param.requires_grad = False
                self.speech_llama_proj.eval()
                logging.info("freeze speech LLAMA proj")
                

        else: 
            logging.info('Loading speech LLAMA proj for tltr module')
            if self.use_tltr :
                self.speech_llama_proj = nn.Linear(
                    1280, self.llama_model.config.hidden_size
                )


        # prepare prompts
        self.prompt_dict = {}
        if prompt_path:
            try:
                raw_prompts = json.load(open(prompt_path, "r"))
            except:
                print("Failed to load prompt! Try to use utf-8 encoding.")
                raw_prompts = json.load(open(prompt_path, "r", encoding='utf-8'))
            for task in raw_prompts.keys():
                if task == 'emotion_prediction_in_conversation' or task == 'only_emotion_prediction_in_conversation' :
                    filted_prompts = [raw_prompt for raw_prompt in raw_prompts[task] if "<DIALOGUE>" in raw_prompt]
                else :
                    filted_prompts = [raw_prompt for raw_prompt in raw_prompts[task] if "<SpeechHere>" in raw_prompt]
                self.prompt_dict[task] = [prompt_template.format(p) for p in filted_prompts]
                # print(self.prompt_dict[task])
            print("Loading training prompts done!")

    def _encode_auditory_feature(self, speech_embeds):
        with self.maybe_autocast():
            if self.use_tltr and not self.use_speech_Qformer :
                speech_embeds = self.tltr_encoder(speech_embeds)
                speech_embeds = self.ln_speech(speech_embeds)
                speech_embeds_output = self.speech_llama_proj(speech_embeds)
                speech_atts = torch.ones(speech_embeds_output.size()[:-1], dtype=torch.long).to(speech_embeds_output.device)
                query_output = None
            elif self.use_speech_Qformer:
                if self.use_whisper : 
                    if self.use_tltr :
                        speech_embeds = self.tltr_encoder(speech_embeds)
                        speech_embeds = self.ln_speech(speech_embeds)
                        
                    else : 
                        speech_embeds = self.ln_speech(speech_embeds)
                
                
                speech_atts = torch.ones(speech_embeds.size()[:-1], dtype=torch.long).to(speech_embeds.device)
                
                query_tokens = self.speech_query_tokens.expand(speech_embeds.shape[0], -1, -1)

                query_output = self.speech_Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=speech_embeds,
                    encoder_attention_mask=speech_atts,
                    return_dict=True,
                )
                speech_embeds_output = self.speech_llama_proj(query_output.last_hidden_state)
                speech_atts = torch.ones(speech_embeds_output.size()[:-1], dtype=torch.long).to(speech_embeds_output.device)
            else:
                raise NotImplementedError

        return speech_embeds_output, speech_atts, query_output, speech_embeds
    
    def chunk_level_pooling(self, target_embeds, num_pools=10):  
        batch_size, seq_length, feature_dim = target_embeds.size()
        pool_size = seq_length // num_pools 
        
    
        pooled_embeds = torch.zeros(batch_size, num_pools, feature_dim)
        
        for i in range(num_pools):
            start_idx = i * pool_size
            end_idx = start_idx + pool_size if i < num_pools - 1 else seq_length
            pooled_embeds[:, i, :] = target_embeds[:, start_idx:end_idx, :].mean(dim=1)
        
        return pooled_embeds
    
    def pad_tensor(self, tensor, pad_size):
        # tensor: [batch_size, sequence_length, feature_dimension] or [sequence_length, feature_dimension]
        if tensor.dim() == 3:
            padding = pad_size - tensor.size(1)
            if padding > 0:
                return F.pad(tensor, (0, 0, 0, padding), value=0)
            else:
                return tensor
        elif tensor.dim() == 2:
            padding = pad_size - tensor.size(1)
            if padding > 0:
                return F.pad(tensor, (0, padding), value=0)
            else:
                return tensor
        else:
            raise ValueError("Unsupported tensor dimension: {}".format(tensor.dim()))
        
    def pad_tensor_1(self, tensor, pad_size): 
        # tensor: [batch_size, sequence_length, feature_dimension] or [sequence_length, feature_dimension]
        if tensor.dim() == 3:
            padding = pad_size - tensor.size(1)
            if padding > 0:
                return F.pad(tensor, (0, 0, 0, padding), value=1)
            else:
                return tensor
        elif tensor.dim() == 2:
            padding = pad_size - tensor.size(1)
            if padding > 0:
                return F.pad(tensor, (0, padding), value=1)
            else:
                return tensor
        else:
            raise ValueError("Unsupported tensor dimension: {}".format(tensor.dim()))

    def encode_speech(self, spectrogram, raw_wav=None, emotion_padding_mask=None):
        with self.maybe_autocast():
            speech_embeds = None
            
            if self.use_whisper :
                if self.use_tltr :
                    audio_rep_tuple = self.speech_encoder(spectrogram, output_hidden_states=True, return_dict=True).hidden_states
                    
                    speech_embeds = [
                        torch.tensor(
                            skimage.measure.block_reduce(
                                audio_rep.detach().cpu().numpy(),
                                (1, self.tltr_downsampling_rate, 1),
                                np.mean
                            )
                        ).to(spectrogram.device)
                        for i, audio_rep in enumerate(audio_rep_tuple) if i != 0
                    ]
                                    
                    speech_embeds = torch.stack(speech_embeds, dim=1) # [B, 32, 25, 1280] #tltr input
                    
                else : 
                    speech_embeds = self.speech_encoder(spectrogram, return_dict=True).last_hidden_state 
            

        return self._encode_auditory_feature(speech_embeds)

    def prompt_wrap(self, embeds, atts, prompt, multi_prompt=False):
        lora_text_mask = None
        lora_emotion_mask = None
       
        
        if prompt:
            if multi_prompt:
                p_before = []
                p_after = []
                for i, p in enumerate(prompt):
                    b, a = p.split("<SpeechHere>")
                    p_before.append(b)
                    p_after.append(a)
                
                p_before_tokens = self.llama_tokenizer(
                    p_before, return_tensors="pt", add_special_tokens=False
                ).to(embeds.device)
                p_before_embeds = self.llama_model.model.embed_tokens(p_before_tokens.input_ids) if not self.lora else self.llama_model.model.model.embed_tokens(p_before_tokens.input_ids)

                p_after_tokens = self.llama_tokenizer(
                    p_after, return_tensors="pt", padding="longest", add_special_tokens=False
                ).to(embeds.device)
                p_after_embeds = self.llama_model.model.embed_tokens(p_after_tokens.input_ids) if not self.lora else self.llama_model.model.model.embed_tokens(p_after_tokens.input_ids)

                wrapped_embeds = torch.cat([p_before_embeds, embeds, p_after_embeds], dim=1)
                wrapped_atts = torch.cat([p_before_tokens.attention_mask, atts, p_after_tokens.attention_mask], dim=1)
            else:
                batch_size = embeds.shape[0]
                p_before, p_after = prompt.split("<SpeechHere>")

                p_before_tokens = self.llama_tokenizer(
                    p_before, return_tensors="pt", add_special_tokens=False
                ).to(embeds.device)
                p_after_tokens = self.llama_tokenizer(
                    p_after, return_tensors="pt", add_special_tokens=False
                ).to(embeds.device)
                p_before_embeds = self.llama_model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1) if not self.lora else self.llama_model.model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
                p_after_embeds = self.llama_model.model.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1) if not self.lora else self.llama_model.model.model.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)

                wrapped_embeds = torch.cat([p_before_embeds, embeds, p_after_embeds], dim=1)
                wrapped_atts = torch.cat([p_before_tokens.attention_mask, atts, p_after_tokens.attention_mask], dim=1)
                
            if self.use_text_plora :
                lora_text_mask = torch.cat([torch.ones_like(p_before_tokens.attention_mask),
                                torch.zeros_like(atts),
                                torch.ones_like(p_after_tokens.attention_mask)], dim=1)
            if self.use_emotion_plora :
                lora_emotion_mask = torch.cat([torch.zeros_like(p_before_tokens.attention_mask),
                                torch.ones_like(atts),
                                torch.zeros_like(p_after_tokens.attention_mask)], dim=1)

            return wrapped_embeds, wrapped_atts, lora_text_mask, lora_emotion_mask
        else:
            return embeds, atts, lora_text_mask, lora_emotion_mask
        
    def dialogue_prompt_wrap(self, embeds_history, atts_history, transcript, emotion, speaker, prompt, multi_prompt=False):
        lora_text_mask = None
        lora_emotion_mask = None
        batch_lora_text_mask = None
        batch_lora_emotion_mask = None
        wrapped_lora_text_mask = None
        wrapped_lora_emotion_mask = None
        
        if prompt:
            batch_size = len(speaker) 
            batch_embeds = []
            batch_atts = []
            
            if self.use_text_plora :
                batch_lora_text_mask = []
            if self.use_emotion_plora :
                batch_lora_emotion_mask = []
            
            max_len_embeds = 0
            max_len_atts = 0
            
            for dialogue_idx in range(batch_size) :
                revise_prompt = False
                if self.use_emotion_label :
                    emotion_hist = emotion[dialogue_idx].split('|')
                else :
                    embeds_hist = embeds_history[dialogue_idx]
                    atts_hist = atts_history[dialogue_idx]
                
                
                transcript_hist = transcript[dialogue_idx].split('|')
                speaker_hist = speaker[dialogue_idx].split('|')
                
                tgt_sen = transcript_hist[-1]
                tgt_spk = speaker_hist[-1]
                
                prompt[dialogue_idx] = prompt[dialogue_idx].replace('<TGT_SEN>', tgt_sen)
                prompt[dialogue_idx] = prompt[dialogue_idx].replace('<TGT_SPK>', tgt_spk)
                
                d_before , _ = prompt[dialogue_idx].split("<DIALOGUE>")
                                
                template = "speaker <SPK> (says with <Emotion><EmotionHere></Emotion>): <SEN>\n"
                template_target = "speaker <SPK> (says): <SEN>\n"
                
                dialogue_embeds = []
                dialogue_atts = []
                
                if self.use_text_plora :
                    lora_text_mask = []
                if self.use_emotion_plora :
                    lora_emotion_mask = []

                if self.use_emotion_label : # emotion label
                    for utt_idx in range(len(speaker_hist)):
                        if utt_idx != (len(speaker_hist) - 1) :             
                            temp_template = template.replace('<SPK>', speaker_hist[utt_idx])
                            temp_template = temp_template.replace('<SEN>', transcript_hist[utt_idx])
                            
                            if utt_idx == 0 :
                                temp_template = d_before + temp_template # d_before => 'USER: '
                                
                            temp_template = temp_template.replace('<Emotion><EmotionHere></Emotion>', emotion_hist[utt_idx])
                            
                            if emotion_hist[utt_idx] != 'neutral' and len(emotion_hist[utt_idx].split(' ')) == 1 :
                                revise_prompt = True
                                
                            utt_tokens = self.llama_tokenizer(
                            temp_template, return_tensors="pt", add_special_tokens=False
                            ).to(torch.device("cuda")) 
                            
                            utt_embeds = self.llama_model.model.embed_tokens(utt_tokens.input_ids) if not self.lora else self.llama_model.model.model.embed_tokens(utt_tokens.input_ids)

                            dialogue_embeds.append(utt_embeds)
                            
                            dialogue_atts.append(utt_tokens.attention_mask)
                            
                            if self.use_text_plora :
                                lora_text_mask.append(torch.ones_like(utt_tokens.attention_mask))
                            if self.use_emotion_plora :
                                lora_emotion_mask.append(torch.zeros_like(utt_tokens.attention_mask))
                                
                            print(temp_template, end="")
                            
                        else :
                            temp_template_target = template_target.replace('<SPK>', speaker_hist[utt_idx])
                            temp_template_target = temp_template_target.replace('<SEN>', transcript_hist[utt_idx])
                            p_tokens = self.llama_tokenizer(
                            temp_template_target, return_tensors="pt", add_special_tokens=False
                            ).to(torch.device("cuda")) 
                            p_embeds = self.llama_model.model.embed_tokens(p_tokens.input_ids) if not self.lora else self.llama_model.model.model.embed_tokens(p_tokens.input_ids)
                            
                            dialogue_embeds.append(p_embeds)
                            
                            dialogue_atts.append(p_tokens.attention_mask)
                            
                            if self.use_text_plora :
                                lora_text_mask.append(torch.ones_like(p_tokens.attention_mask))
                            if self.use_emotion_plora :
                                lora_emotion_mask.append(torch.zeros_like(p_tokens.attention_mask))
                            print(temp_template_target, end="")
                            
                else : 
                    for utt_idx in range(len(speaker_hist)):
                        if utt_idx != (len(speaker_hist) - 1) :             
                            temp_template = template.replace('<SPK>', speaker_hist[utt_idx])
                            temp_template = temp_template.replace('<SEN>', transcript_hist[utt_idx])
                            
                            if utt_idx == 0 :
                                temp_template = d_before + temp_template 
                                
                            p_before, p_after = temp_template.split("<EmotionHere>")
                            p_before_tokens = self.llama_tokenizer(
                            p_before, return_tensors="pt", add_special_tokens=False
                            ).to(embeds_hist[utt_idx].device)
                            p_after_tokens = self.llama_tokenizer(
                                p_after, return_tensors="pt", add_special_tokens=False
                            ).to(embeds_hist[utt_idx].device)
                            p_before_embeds = self.llama_model.model.embed_tokens(p_before_tokens.input_ids) if not self.lora else self.llama_model.model.model.embed_tokens(p_before_tokens.input_ids)
                            p_after_embeds = self.llama_model.model.embed_tokens(p_after_tokens.input_ids) if not self.lora else self.llama_model.model.model.embed_tokens(p_after_tokens.input_ids)                      

                            dialogue_embeds.append(p_before_embeds)
                            dialogue_embeds.append(embeds_hist[utt_idx])
                            dialogue_embeds.append(p_after_embeds)
                            
                            dialogue_atts.append(p_before_tokens.attention_mask)
                            dialogue_atts.append(atts_hist[utt_idx])
                            dialogue_atts.append(p_after_tokens.attention_mask)
                            
                            if self.use_text_plora :
                                lora_text_mask.append(torch.ones_like(p_before_tokens.attention_mask))
                                lora_text_mask.append(torch.zeros_like(atts_hist[utt_idx]))
                                lora_text_mask.append(torch.ones_like(p_after_tokens.attention_mask))
                            if self.use_emotion_plora :
                                lora_emotion_mask.append(torch.zeros_like(p_before_tokens.attention_mask))
                                lora_emotion_mask.append(torch.ones_like(atts_hist[utt_idx]))
                                lora_emotion_mask.append(torch.zeros_like(p_after_tokens.attention_mask))
                                
                            print(temp_template, end="")
                            
                        else :
                            temp_template_target = template_target.replace('<SPK>', speaker_hist[utt_idx])
                            temp_template_target = temp_template_target.replace('<SEN>', transcript_hist[utt_idx])
                            p_tokens = self.llama_tokenizer(
                            temp_template_target, return_tensors="pt", add_special_tokens=False
                            ).to(embeds_hist[utt_idx].device)
                            p_embeds = self.llama_model.model.embed_tokens(p_tokens.input_ids) if not self.lora else self.llama_model.model.model.embed_tokens(p_tokens.input_ids)
                            
                            dialogue_embeds.append(p_embeds)
                            
                            dialogue_atts.append(p_tokens.attention_mask)
                            
                            if self.use_text_plora :
                                lora_text_mask.append(torch.ones_like(p_tokens.attention_mask))
                            if self.use_emotion_plora :
                                lora_emotion_mask.append(torch.zeros_like(p_tokens.attention_mask))
                            print(temp_template_target, end="")

                
                post_target = prompt[dialogue_idx].split("<DIALOGUE>")[1]
                if revise_prompt :
                    post_target = post_target.replace(' and intensity','') 
                
                print(post_target)
                d_after = post_target
                if self.use_emotion_label :
                    d_after_tokens = self.llama_tokenizer(
                    d_after, return_tensors="pt", add_special_tokens=False
                    ).to(torch.device("cuda")) 
                else :
                    d_after_tokens = self.llama_tokenizer(
                    d_after, return_tensors="pt", add_special_tokens=False
                    ).to(embeds_hist[-1].device)
                d_after_embeds = self.llama_model.model.embed_tokens(d_after_tokens.input_ids) if not self.lora else self.llama_model.model.model.embed_tokens(d_after_tokens.input_ids)
                
                dialogue_embeds.append(d_after_embeds)
                dialogue_atts.append(d_after_tokens.attention_mask)
                
                if self.use_text_plora :
                    lora_text_mask.append(torch.ones_like(d_after_tokens.attention_mask))
                    lora_text_mask = torch.cat(lora_text_mask, dim=1)
                    batch_lora_text_mask.append(lora_text_mask)
                    
                if self.use_emotion_plora :
                    lora_emotion_mask.append(torch.zeros_like(d_after_tokens.attention_mask))
                    lora_emotion_mask = torch.cat(lora_emotion_mask, dim=1)
                    batch_lora_emotion_mask.append(lora_emotion_mask)
                    
                dialogue_embeds_final = torch.cat(dialogue_embeds, dim=1)
                dialogue_atts_final = torch.cat(dialogue_atts, dim=1)
                
                batch_embeds.append(dialogue_embeds_final)
                batch_atts.append(dialogue_atts_final)
            
            max_len_embeds = max(tensor.size(1) for tensor in batch_embeds)
            max_len_atts = max(tensor.size(1) for tensor in batch_atts)
            
            batch_embeds_padded = [self.pad_tensor(embed, max_len_embeds) for embed in batch_embeds]
            batch_atts_padded = [self.pad_tensor(att, max_len_atts) for att in batch_atts]
            
            if self.use_text_plora :
                batch_lora_text_mask_padded = [self.pad_tensor_1(lora_text_mask, max_len_atts) for lora_text_mask in batch_lora_text_mask] 
                wrapped_lora_text_mask = torch.stack(batch_lora_text_mask_padded, dim=0)
                wrapped_lora_text_mask = wrapped_lora_text_mask.squeeze(1)

            if self.use_emotion_plora :
                batch_lora_emotion_mask_padded = [self.pad_tensor(lora_emotion_mask, max_len_atts) for lora_emotion_mask in batch_lora_emotion_mask] 
                wrapped_lora_emotion_mask = torch.stack(batch_lora_emotion_mask_padded, dim=0)
                wrapped_lora_emotion_mask = wrapped_lora_emotion_mask.squeeze(1)
                

            wrapped_embeds = torch.stack(batch_embeds_padded, dim=0)
            wrapped_atts = torch.stack(batch_atts_padded, dim=0)

            wrapped_embeds = wrapped_embeds.squeeze(1)
            wrapped_atts = wrapped_atts.squeeze(1)
            
            return wrapped_embeds, wrapped_atts, wrapped_lora_text_mask, wrapped_lora_emotion_mask
        else: 
            samples_embeds = [torch.stack(embeds_hist, dim=0)for embeds_hist in embeds_history] 
            cat_embeds = torch.stack(samples_embeds, dim=0) 
            
            samples_atts = [torch.stack(atts_hist, dim=0)for atts_hist in atts_history]
            cat_atts = torch.stack(samples_atts, dim=0)
                
            return cat_embeds, cat_atts
        
    def update_lora_mask(self, target_token_mask, inference_mode: bool, target_adapter_name): 
        if not self.use_plora:
            return
        
        self.llama_model.update_inference_mode(inference_mode)
        self.llama_model.update_lora_mask(target_adapter_name, target_token_mask)
        

    def forward(self, samples, verbose=False):
                
        task = list(set(samples["task"]))
        if len(task) > 1 or "QA" in task:
            self.multi_prompt = True
            
        if "emotion_prediction_in_conversation" in task or 'only_emotion_prediction_in_conversation'in task:
            self.dialogue_prompt = True

        if self.prompt_dict:
            if self.multi_prompt:
                prompt = [random.choice(self.prompt_dict[task]) for task in samples["task"]]

            else:
                prompt = random.choice(self.prompt_dict[samples["task"][0]])
                

        if self.dialogue_prompt:
            if self.use_extracted_feature : 
                transcript = samples["transcript"]
                speaker = samples["speaker"]
                speech_embeds_history = [] 
                speech_atts_history = [] 
                emotion = None
                
                if self.use_emotion_label : 
                    emotion = samples["emotion"]
                else : 
                    if self.use_whisper : 
                        whisper_features_history = samples.get("whisper_features_history", None)
                        
                        speech_embeds_history = []
                        speech_atts_history = [] 
                        
                        for whisper_features_hist in whisper_features_history : 
                            speech_embeds_hist = []
                            speech_atts_hist = []
                            
                            for whisper_features in whisper_features_hist :
                                
                                speech_embeds, speech_atts, query_output, _ = self._encode_auditory_feature(whisper_features)
                                
                                speech_embeds_hist.append(speech_embeds)
                                speech_atts_hist.append(speech_atts)
                                
                            speech_embeds_history.append(speech_embeds_hist)
                            speech_atts_history.append(speech_atts_hist)            
            else : 
                transcript = samples["transcript"]
                speaker = samples["speaker"]
            
                speech_embeds_history = [] 
                speech_atts_history = [] 
                emotion = None
                
                if self.use_emotion_label :
                    emotion = samples["emotion"]
                else :
                    spectrogram_history = samples.get("spectrogram_history", None)
                    audio_history = samples.get("audio_history", None)
                    
                    if self.use_whisper : 
                        for spectrogram_hist in spectrogram_history : 
                            speech_embeds_hist = []
                            speech_atts_hist = []
                                
                            for spectrogram in spectrogram_hist :
                                spectrogram = spectrogram.unsqueeze(0) 
                                speech_embeds, speech_atts, query_output, _ = self.encode_speech(spectrogram, raw_wav=None)
                                speech_embeds_hist.append(speech_embeds)
                                speech_atts_hist.append(speech_atts)
                                
                            speech_embeds_history.append(speech_embeds_hist)
                            speech_atts_history.append(speech_atts_hist)
                
            if self.prompt_dict:
                speech_embeds, speech_atts, lora_text_mask, lora_emotion_mask = self.dialogue_prompt_wrap(speech_embeds_history, speech_atts_history, transcript, emotion, speaker, prompt, multi_prompt=self.multi_prompt)

        else :
            if self.use_extracted_feature : 
                whisper_features = samples["whisper_features"]
                

                speech_embeds, speech_atts, query_output, speech_embeds_pre = self._encode_auditory_feature(whisper_features)

            else :
                
                spectrogram = samples["spectrogram"]
                raw_wav = samples.get("raw_wav", None)

                speech_embeds, speech_atts, query_output, speech_embeds_pre = self.encode_speech(spectrogram, raw_wav=raw_wav)
            
            
            
            if self.prompt_dict:
                speech_embeds, speech_atts, lora_text_mask, lora_emotion_mask = self.prompt_wrap(speech_embeds, speech_atts, prompt, multi_prompt=self.multi_prompt)
                
        
        text = [t + self.end_sym for t in samples["text"]]
        print(f'GT => {text}\n')
        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(speech_atts.device) 
        to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids) if not self.lora else self.llama_model.model.model.embed_tokens(to_regress_tokens.input_ids)
        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )
        empty_targets = (
            torch.ones(
                [speech_atts.shape[0], speech_atts.shape[1] + 1],
                dtype=torch.long
            ).to(speech_atts.device).fill_(-100) 
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        batch_size = speech_embeds.shape[0]
        bos = torch.ones(
            [batch_size, 1],
            dtype=to_regress_tokens.input_ids.dtype,
            device=to_regress_tokens.input_ids.device,
        ) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.llama_model.model.embed_tokens(bos) if not self.lora else self.llama_model.model.model.embed_tokens(bos)
        atts_bos = speech_atts[:, :1]

        inputs_embeds = torch.cat([bos_embeds, speech_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, speech_atts, to_regress_tokens.attention_mask], dim=1)
        
        if self.use_text_plora :
            lora_text_mask = torch.cat([torch.ones_like(atts_bos),
                                                lora_text_mask,
                                                torch.ones_like(to_regress_tokens.attention_mask)], dim=1)
            self.update_lora_mask(lora_text_mask, False, "text")
            
            
            
        if self.use_emotion_plora : 
            lora_emotion_mask = torch.cat([torch.zeros_like(atts_bos),
                                            lora_emotion_mask,
                                            torch.zeros_like(to_regress_tokens.attention_mask)], dim=1)
            self.update_lora_mask(lora_emotion_mask, False, "emotion")

        
        # calulate loss
        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
            loss = outputs.loss
            
        if verbose:
            nvocab = self.llama_model.config.vocab_size
            results = outputs.logits[:, empty_targets.size(1) - 1: -1, :].contiguous().view(-1, nvocab).argmax(dim=-1)
            labels = targets[:, empty_targets.size(1):].contiguous().view(-1)
            mask = (labels != -100)
            correct = (results[mask] == labels[mask]).float().sum()
            total = len(labels[mask])

        if verbose:
            return {"loss": loss, "correct": correct, "total": total}

        return {"loss": loss}
            
        

    def generate(self, samples, generate_cfg, prompts=None):
        if self.dialogue_prompt : 
            if self.use_extracted_feature :  
                transcript = samples["transcript"]
                
                speaker = samples["speaker"]
                speech_embeds_history = []
                speech_atts_history = [] 
                emotion = []
                
                if self.use_emotion_label : 
                    emotion = samples["emotion"]
                else :
                    whisper_features_history = samples.get("whisper_features_history", None)
                
                    if self.use_whisper :
                        for whisper_features_hist in whisper_features_history : 
                            speech_embeds_hist = []
                            speech_atts_hist = []
                            
                            for whisper_features in whisper_features_hist :
                                speech_embeds, speech_atts, query_output, _ = self._encode_auditory_feature(whisper_features)
                                
                                speech_embeds_hist.append(speech_embeds)
                                speech_atts_hist.append(speech_atts)
                                
                            speech_embeds_history.append(speech_embeds_hist)
                            speech_atts_history.append(speech_atts_hist)
                        
            else :  
                transcript = samples["transcript"]
                speaker = samples["speaker"]
                emotion = []
                
                speech_embeds_history = [] 
                speech_atts_history = [] 
                
                if self.use_emotion_label : 
                    emotion = samples["emotion"]
                
                else :
                    spectrogram_history = samples.get("spectrogram_history", None)
                    audio_history = samples.get("audio_history", None)
                    emotion_padding_mask_history = samples.get("padding_mask_history", None)

                    if self.use_whisper : 
                        for spectrogram_hist in spectrogram_history : 
                            speech_embeds_hist = []
                            speech_atts_hist = []
                                
                            for spectrogram in spectrogram_hist :
                                spectrogram = spectrogram.unsqueeze(0) 
                                speech_embeds, speech_atts, query_output, _ = self.encode_speech(spectrogram, raw_wav=None)
                                speech_embeds_hist.append(speech_embeds)
                                speech_atts_hist.append(speech_atts)
                                
                            speech_embeds_history.append(speech_embeds_hist)
                            speech_atts_history.append(speech_atts_hist)
                
            if prompts is not None :
                speech_embeds, speech_atts, lora_text_mask, lora_emotion_mask = self.dialogue_prompt_wrap(speech_embeds_history, speech_atts_history, transcript, emotion, speaker, prompts, multi_prompt=True)
        else : 
            if self.use_extracted_feature : 
                whisper_features = samples["whisper_features"]
                speech_embeds, speech_atts, query_output, _ = self._encode_auditory_feature(whisper_features)
                
            else : 
                batch_size = samples["spectrogram"].shape[0]
                spectrogram = samples["spectrogram"]
                raw_wav = samples.get("raw_wav", None)

                speech_embeds, speech_atts, query_output, _ = self.encode_speech(spectrogram, raw_wav=raw_wav)
            
            if prompts is not None:
                speech_embeds, speech_atts, lora_text_mask, lora_emotion_mask = self.prompt_wrap(speech_embeds, speech_atts, prompts, multi_prompt=True)

        batch_size = speech_embeds.shape[0]
        
        bos = torch.ones(
            [batch_size, 1],
            dtype=torch.int32,
            device=speech_embeds.device,
        ) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.llama_model.model.embed_tokens(bos) if not self.lora else self.llama_model.model.model.embed_tokens(bos)
        atts_bos = speech_atts[:, :1]

        embeds = torch.cat([bos_embeds, speech_embeds], dim=1)
        attns = torch.cat([atts_bos, speech_atts], dim=1)
        
        if self.use_text_plora :
            lora_text_mask = torch.cat([torch.ones_like(atts_bos),
                                    lora_text_mask], dim=1)
            self.update_lora_mask(lora_text_mask, True, "text")
        
        if self.use_emotion_plora :
            lora_emotion_mask = torch.cat([torch.zeros_like(atts_bos),
                                            lora_emotion_mask], dim=1)
            self.update_lora_mask(lora_emotion_mask, True, "emotion")
            
        stop_words_ids = [torch.tensor([2]).cuda()]  
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
        outputs = self.llama_model.generate(
            inputs_embeds=embeds,
            max_new_tokens=generate_cfg.get("max_new_tokens", 200),
            stopping_criteria=stopping_criteria,
            num_beams=generate_cfg.get("num_beams", 4),
            do_sample=generate_cfg.get("do_sample", False),
            min_length=generate_cfg.get("min_length", 1),
            temperature=generate_cfg.get("temperature", 1.0),
            top_p=generate_cfg.get("top_p", 0.9),
            repetition_penalty=generate_cfg.get("repetition_penalty", 1.0),
            length_penalty=generate_cfg.get("length_penalty", 1.0),
            attention_mask=attns,
        )
        text = self.llama_tokenizer.batch_decode(outputs, add_special_tokens=False)

        return text
    
    def load_checkpoint(self, ckpt_path, filter_keys, exclude_keys=None):
        if not ckpt_path:
            logging.info("Checkpoint path not provided.")
            return {}
      
        print("Load JELLY checkpoint from: {}".format(ckpt_path))
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = {k: v for k, v in ckpt['model'].items() if any(key in k for key in filter_keys)}
        if exclude_keys:
            state_dict = {k: v for k, v in state_dict.items() if not any(key in k for key in exclude_keys)}
      
        print("Keys in filtered state_dict: {}".format(state_dict.keys()))
        
        return state_dict

    @classmethod
    def from_config(cls, config):
        whisper_config_d_model = config.get("whisper_config_d_model", 1280)
        tltr_embed_dim = config.get("tltr_embed_dim", 1280)
        tltr_downsampling_rate = config.get("tltr_downsampling_rate", 40)
        use_extracted_feature = config.get("use_extracted_feature")
        use_whisper = config.get("use_whisper")
        use_tltr = config.get("use_tltr", False)
        use_pretrained_proj = config.get("use_pretrained_proj", True)
        llama_path = config.get("llama_path")
        whisper_path = config.get("whisper_path")
        freeze_whisper = config.get("freeze_whisper", True)
        freeze_tltr = config.get("freeze_tltr", False)
        freeze_lora = config.get("freeze_lora", False)
        freeze_text_plora = config.get("freeze_text_plora", False)
        freeze_emotion_plora = config.get("freeze_emotion_plora", False)

        use_speech_Qformer = config.get("use_speech_Qformer", True)
        num_speech_query_token = config.get("num_speech_query_token", 25)
        freeze_speech_QFormer = config.get("freeze_speech_QFormer", False)

        speech_llama_proj_model = config.get("speech_llama_proj_model", "")
        freeze_speech_llama_proj = config.get("freeze_speech_llama_proj", False)

        lora = config.get("lora", True)
        lora_rank = config.get("lora_rank", 8)
        lora_alpha = config.get("lora_alpha", 32)
        lora_dropout = config.get("lora_dropout", 0.1)
        
        use_plora = config.get("use_plora", False)
        
        use_text_plora = config.get("use_text_plora", False)
        text_plora_rank = config.get("text_plora_rank", 8)
        text_plora_alpha = config.get("text_plora_alpha", 32)
        text_plora_dropout = config.get("text_plora_dropout", 0.1)
        text_plora_target_modules = config.get("text_plora_target_modules", "")
        
        use_emotion_plora = config.get("use_emotion_plora", False)
        emotion_plora_rank = config.get("emotion_plora_rank", 8)
        emotion_plora_alpha = config.get("emotion_plora_alpha", 32)
        emotion_plora_dropout = config.get("emotion_plora_dropout", 0.1)
        emotion_plora_target_modules = config.get("emotion_plora_target_modules", "")
        

        target_speech_vocab_size = config.get("target_speech_vocab_size", 1024)
        use_input_embeds = config.get("use_input_embeds", False)
        emb_dim = config.get("emb_dim", 256)
        multi_prompt = config.get("multi_prompt", False)
        dialogue_prompt = config.get("dialogue_prompt", False)
        use_emotion_label = config.get("use_emotion_label", False)
        prompt_path = config.get("prompt_path", "")
        prompt_template = config.get("prompt_template", "")
        max_txt_len = config.get("max_txt_len", 128)
        end_sym = config.get("end_sym", "</s>")
        low_resource = config.get("low_resource", False)
        device_8bit = config.get("device_8bit", 0)

        model = cls(
            whisper_config_d_model=whisper_config_d_model,
            tltr_embed_dim=tltr_embed_dim,
            tltr_downsampling_rate=tltr_downsampling_rate,
            use_extracted_feature=use_extracted_feature,
            use_whisper=use_whisper,
            use_tltr=use_tltr,
            llama_path=llama_path,
            whisper_path=whisper_path,
            freeze_whisper=freeze_whisper,
            freeze_tltr=freeze_tltr,
            freeze_lora=freeze_lora,
            freeze_text_plora = freeze_text_plora,
            freeze_emotion_plora = freeze_emotion_plora,
            use_speech_Qformer=use_speech_Qformer,
            num_speech_query_token=num_speech_query_token,
            freeze_speech_QFormer=freeze_speech_QFormer,
            speech_llama_proj_model=speech_llama_proj_model,
            freeze_speech_llama_proj=freeze_speech_llama_proj,
            lora=lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            use_plora=use_plora,
            use_text_plora=use_text_plora,
            text_plora_rank=text_plora_rank,
            text_plora_alpha=text_plora_alpha, 
            text_plora_dropout=text_plora_dropout,
            text_plora_target_modules=text_plora_target_modules,
            use_emotion_plora=use_emotion_plora,
            emotion_plora_rank=emotion_plora_rank,
            emotion_plora_alpha=emotion_plora_alpha,
            emotion_plora_dropout=emotion_plora_dropout,
            emotion_plora_target_modules=emotion_plora_target_modules,
            target_speech_vocab_size=target_speech_vocab_size,
            use_input_embeds=use_input_embeds,
            emb_dim=emb_dim,
            multi_prompt=multi_prompt,
            dialogue_prompt=dialogue_prompt,
            use_emotion_label=use_emotion_label,
            prompt_path=prompt_path,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            low_resource=low_resource,
            device_8bit=device_8bit,
        )
                
        qformer_ckpt_path = config.get("qformer_ckpt", "")

        complete_state_dict = {}

        qformer_filter_keys = ['ln_emotion', 'ln_speech', 'speech_query_tokens', 'speech_Qformer']

        if use_tltr:
            tltr_ckpt_path = config.get("tltr_ckpt", "")
            tltr_state_dict = model.load_checkpoint(tltr_ckpt_path, ['tltr_encoder'])
            complete_state_dict.update(tltr_state_dict)

        if use_pretrained_proj:
            qformer_filter_keys.append('speech_llama_proj')
            qformer_state_dict = model.load_checkpoint(qformer_ckpt_path, qformer_filter_keys)
        else:
            qformer_state_dict = model.load_checkpoint(qformer_ckpt_path, qformer_filter_keys, exclude_keys=['speech_llama_proj'])

        if qformer_state_dict:
            complete_state_dict.update(qformer_state_dict)
        else:
            logging.info("Not Loading JELLY qformer_ckpt!")

        if use_plora:
            paths = {
                "text": config.get("text_plora_ckpt", ""),
                "emotion": config.get("emotion_plora_ckpt", ""),
            }

            for key, path in paths.items():
                if path:
                    filter_keys = [f'lora_A.{key}', f'lora_B.{key}', 'rotary_emb']
                    state_dict = model.load_checkpoint(path, filter_keys)
                    complete_state_dict.update(state_dict)

        else:
            lora_ckpt_path = config.get("lora_ckpt", "")
            if lora_ckpt_path:
                lora_state_dict = model.load_checkpoint(lora_ckpt_path, ['llama_model'])
                complete_state_dict.update(lora_state_dict)

        if complete_state_dict:
            model.load_state_dict(complete_state_dict, strict=False)
            logging.info("Model state dict loaded successfully")
        else:
            logging.info("No state dict to load into the model")

       
        return model
