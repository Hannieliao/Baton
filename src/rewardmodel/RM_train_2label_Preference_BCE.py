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
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda:0')
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

# human feedback data
with open("grouped_ratings.json", "r") as f:
    data = json.load(f)
    
prompts_image_features = data["grouped_ratings"]

# Create the dataset and data loader
preference_dataset = PreferenceDataset(prompts_image_features)
data_loader = DataLoader(preference_dataset, batch_size=1, shuffle=True)

# Initialize the model, loss function, and optimizer
model = RewardModel().to(device)             
# criterion = nn.BCEWithLogitsLoss()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
EPOCHS = 101

losses = 0
average_losses_per_epoch = []

# Training loop
for epoch in tqdm(range(EPOCHS)):  # Let's train for 5 epochs
    for (preferred, non_preferred), label in data_loader:
        # import pdb; pdb.set_trace()
        # preferred = preferred.to(device)
        # non_preferred = non_preferred.to(device)
        # Forward pass
        preferred = os.path.join("/home/hhn/tango/human/static", preferred[0])
        non_preferred = os.path.join("/home/hhn/tango/human/static", non_preferred[0])
        
        enc_preferred = encode_data(preferred)
        enc_non_preferred = encode_data(non_preferred)
        
        pred_preferred = model(enc_preferred)
        pred_non_preferred = model(enc_non_preferred)

        # pred_diff = pred_preferred - pred_non_preferred
        # label = label.to(device)
        # pred_diff = pred_diff.to(device)

        # Compute loss 
        # loss = criterion(pred_diff[0], label.float())
        loss = criterion(pred_preferred, torch.ones_like(pred_preferred)) + criterion(pred_non_preferred, torch.zeros_like(pred_non_preferred))
        losses += (loss.item())

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            checkpoint = {
                'epoch': epoch,
                'rewardmodel_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }
            # torch.save(model.state_dict(), f"model_weights_epoch{epoch}.pth")#保存权重
            torch.save(checkpoint, f'rm_ckpt_CLAP_BCE_2label_Preference/model_weights_epoch{epoch}.pth')
    
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')
    average_losses_per_epoch.append(loss.item())
    
# Let's print out the model parameters
for name, param in model.named_parameters():
    print(f"{name}: {param.data}")
    
plt.plot(average_losses_per_epoch)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("loss")
plt.savefig("loss_CLAP_BCE_2label_Preference.png")
plt.show()
