# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
import typing as tp

import torch
import torchmetrics
from transformers import RobertaTokenizer  # type: ignore

import numpy as np
import librosa
import glob
import os
import json
import argparse

import sys
sys.path.append('/home/hhn/tango/laionclap')

from audiocraft.data.audio_utils import convert_audio
from audiocraft.environment import AudioCraftEnvironment
from audiocraft.utils.utils import load_clap_state_dict

try:
    import laion_clap  # type: ignore
except ImportError:
    laion_clap = None
    
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class TextConsistencyMetric(torchmetrics.Metric):
    """Text consistency metric measuring consistency between audio and text pairs."""

    def update(self, audio: torch.Tensor, text: tp.List[str], sizes: torch.Tensor, sample_rates: torch.Tensor) -> None:
        raise NotImplementedError("implement how to update the metric from the audio and text pairs.")

    def compute(self):
        raise NotImplementedError("implement how to compute the final metric score.")


class CLAPTextConsistencyMetric(TextConsistencyMetric):
    """Text consistency metric relying on Contrastive Language-Audio Pretraining (CLAP).

    This metric is similar to the MuLan Cycle Consistency from MusicLM (https://arxiv.org/pdf/2301.11325.pdf)
    or the CLAP score used in Make-An-Audio (https://arxiv.org/pdf/2301.12661v1.pdf).

    As a joint audio-text embedding model, a pretrained CLAP model can be used to quantify the
    similarity between audio-text pairs. We compute the CLAP embeddings from the text descriptions as
    well as the generated audio based on them, and define the MCC metric as the average cosine similarity
    between these embeddings.

    Model implementation & pre-trained checkpoints: https://github.com/LAION-AI/CLAP
    """
    def __init__(self, model_path: tp.Union[str, Path], model_arch: str = 'HTSAT-tiny', enable_fusion: bool = False):
        super().__init__()
        if laion_clap is None:
            raise ImportError("Please install CLAP to compute text consistency: 'pip install laion_clap'")
        self.add_state("cosine_sum", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("weight", default=torch.tensor(0.), dist_reduce_fx="sum")
        self._initialize_model(model_path, model_arch, enable_fusion)

    def _initialize_model(self, model_path: tp.Union[str, Path], model_arch: str, enable_fusion: bool):
        model_path = AudioCraftEnvironment.resolve_reference_path(model_path)
        self.tokenize = RobertaTokenizer.from_pretrained('roberta-base')
        self.model = laion_clap.CLAP_Module(enable_fusion=enable_fusion, amodel=model_arch)
        self.model_sample_rate = 48_000
        load_clap_state_dict(self.model, model_path)
        self.model.eval()

    def _tokenizer(self, texts: tp.Union[str, tp.List[str]]) -> dict:
        # we use the default params from CLAP module here as well
        return self.tokenize(texts, padding="max_length", truncation=True, max_length=77, return_tensors="pt")

    def update(self, audio: torch.Tensor, text: tp.List[str], sizes: torch.Tensor, sample_rates: torch.Tensor) -> None:
        """Compute cosine similarity between audio and text pairs and accumulate scores over the dataset."""
        assert audio.size(0) == len(text), "Number of audio and text samples should match"
        assert torch.all(sample_rates == sample_rates[0].item()), "All items in batch should have the same sample rate"
        sample_rate = int(sample_rates[0].item())
        # convert audio batch to 48kHz monophonic audio with no channel dimension: [B, C, T] -> [B, T]
        audio = convert_audio(audio, from_rate=sample_rate, to_rate=self.model_sample_rate, to_channels=1).mean(dim=1)
        audio_embeddings = self.model.get_audio_embedding_from_data(audio, use_tensor=True)
        text_embeddings = self.model.get_text_embedding(text, tokenizer=self._tokenizer, use_tensor=True)
        # cosine similarity between the text and the audio embedding
        cosine_sim = torch.nn.functional.cosine_similarity(audio_embeddings, text_embeddings, dim=1, eps=1e-8)
        self.cosine_sum += cosine_sim.sum(dim=0)
        self.weight += torch.tensor(cosine_sim.size(0))

    def compute(self):
        """Computes the average cosine similarty across all audio/text pairs."""
        assert self.weight.item() > 0, "Unable to compute with total number of comparisons <= 0"  # type: ignore
        return (self.cosine_sum / self.weight).item()  # type: ignore

def parse_args():
    parser = argparse.ArgumentParser(description='Compute CLAP scores for audio files.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the CLAP pre-trained model checkpoint')
    parser.add_argument('--model_arch', type=str, default='HTSAT-tiny', help='CLAP model architecture')
    parser.add_argument('--enable_fusion', action='store_true', help='Enable fusion in CLAP model')
    parser.add_argument('--audio_folder', type=str, required=True, help='Path to the folder containing audio files for evaluation')
    parser.add_argument('--output_json', type=str, required=True, help='Path to save the CLAP scores in JSON format')
    return parser.parse_args()

def get_audio_paths(folder_path):
    audio_paths = []
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            if filename.endswith('.wav'):
                audio_path = os.path.join(dirpath, filename)
                audio_paths.append(audio_path)
    return audio_paths

def main():
    args = parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 初始化度量类实例
    clap_metric = CLAPTextConsistencyMetric(model_path=args.model_path, model_arch=args.model_arch, enable_fusion=True).to(device)

    # 准备音频和文本数据
    audio_paths = get_audio_paths(args.audio_folder)

    results = []

    for audio_file in audio_paths:
        audio_data, _ = librosa.load(audio_file, sr=16000) # sample rate should be 48000
        audio_data = audio_data.reshape(1, -1) # Make it (1,T) or (B,T)
        text_data = [audio_file.split("/")[-1].split("_")[-1].replace(".wav", "")] # tune集
        # text_data = [audio_file.split("/")[-1].replace(".wav", "")] # audioldm集  
        audio_data = torch.tensor([audio_data])
        
        sizes = 1 # 音频大小张量，形状为 [B]
        sizes = torch.tensor([sizes])
        sample_rates = 16000
        sample_rates = torch.tensor([sample_rates]) # 音频采样率张量，形状为 [B]
        
        audio_data = audio_data.to(device)
        sizes = sizes.to(device)
        sample_rates = sample_rates.to(device)

        # 更新度量  
        clap_metric.update(audio_data, text_data, sizes, sample_rates)
        # 计算CLAP分数
        clap_score = clap_metric.compute()
        result = {
            "audio": audio_file.split("/")[-1],
            "clap_score": clap_score
        }
        results.append(result)

    with open(args.output_json, "w") as f:
        json.dump(results, f)

    # 计算json的平均数
    with open(args.output_json, "r") as f:
        data = json.load(f)

    scores = [item["clap_score"] for item in data]
    total = sum(scores)
    count = len(scores)
    average = total / count
    print("CLAP平均值为：", average)

if __name__ == "__main__":
    main()
