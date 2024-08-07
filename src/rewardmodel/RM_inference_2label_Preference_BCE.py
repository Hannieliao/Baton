import os
import json
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from itertools import combinations
import matplotlib.pyplot as plt

import librosa
import laion_clap

# Initialize the CLAP model for encoding
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encode_model = laion_clap.CLAP_Module(enable_fusion=False, device=device)
encode_model.load_ckpt()

# Define the functions for audio data conversion
def int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)

def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype(np.int16)

# Assume audio_files is a list of file paths to audio files
audio_files = [...] # Replace with your actual list of audio file paths

# Function to encode audio and text data using the CLAP model
def encode_data(audio_file):
    with torch.no_grad():
    # Load and preprocess the audio file
        audio_data, _ = librosa.load(audio_file, sr=16000)
        audio_data = audio_data.reshape(1, -1)
        audio_data = torch.from_numpy(int16_to_float32(float32_to_int16(audio_data))).float().to(device)
        
        # Extract audio features
        audio_feature = encode_model.get_audio_embedding_from_data(x=audio_data, use_tensor=True)
        
        # Extract text features
        text_data = audio_file.split("/")[-1].split("_")[1]
        text_data = [text_data]*2 # get_text_embedding requires two parameters
        text_feature = encode_model.get_text_embedding(text_data, use_tensor=True)
        text_feature = text_feature[0:1, :]
        
        feature_concat = torch.cat((audio_feature, text_feature), dim=1).to(device)
        
        # return audio_feature, text_feature
        return feature_concat

class PreferenceDataset(Dataset):
    def __init__(self, prompts_features):
        self.samples = []
        # Iterate over each prompt's set of image scores
        for prompt, images_scores in prompts_features.items():
            # Create all possible pairwise combinations
            for (img1, score1), (img2, score2) in combinations(images_scores.items(), 2):
                if score1 != score2:  # We only care about pairs with different scores
                    preferred, non_preferred = (img1, img2) if score1 > score2 else (img2, img1)
                    # Append the pair and label (1 for preferred)
                    self.samples.append(((preferred, non_preferred), 1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        (preferred, non_preferred), label = self.samples[idx]
        return (preferred, non_preferred), label

# Define Rewardmodel
class RewardModel(nn.Module):
    def __init__(self):
        super(RewardModel, self).__init__()
        self.fc1 = nn.Linear(1024, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# Assuming model and device are already defined from the training code

def inference(audio_file):
    # Load and preprocess the audio file
    audio_data, _ = librosa.load(audio_file, sr=16000)
    audio_data = audio_data.reshape(1, -1)
    audio_data = torch.from_numpy(audio_data).float().to(device)
    
    # Encoding the audio and text data
    encoded_data = encode_data(audio_file)  # Using the previously defined encode_data function
    
    # Forward pass through the model
    with torch.no_grad():
        model.eval()  # Set the model to evaluation mode
        predictions = model(encoded_data)
    
    return predictions

model = RewardModel().to(device)
checkpoint = torch.load("rm_ckpt_CLAP_BCE_2label_Preference/model_weights_epoch50.pth") # Replace your checkpoint path
model.load_state_dict(checkpoint['rewardmodel_state_dict'])

# Usage example
audio_file_path = "Baton/data/Audio/2label_Temporal_RD_Audio"
audio_files = [os.path.join(audio_file_path, f) for f in os.listdir(audio_file_path) if f.endswith('.wav')] #["path_to_audio1.wav", "path_to_audio2.wav", ...]
results = []

for audio_file in audio_files:
    predictions = inference(audio_file)
    result = {
        "audio": audio_file,
        "pre_score": predictions
    }
    results.append(result)

results = [np.array(tensor) for tensor in results]
    
with open ("rm_clap_BCE_2label_Prefernece_RA.json", "w") as f:
    json.dump(results, f, default=lambda x: x.tolist())
