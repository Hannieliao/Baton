import os
import json
import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from itertools import combinations
import matplotlib.pyplot as plt

import librosa
import laion_clap

device = torch.device("cuda:0")

from torch.utils.data import Dataset

class AudioDataset(Dataset):
    def __init__(self, audio_files, encode_model, data):
        self.audio_files = audio_files
        self.encode_model = encode_model
        self.data = data

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_file = self.audio_files[idx]

        audio_data, _ = librosa.load(audio_file, sr=16000)
        audio_data = audio_data.reshape(1, -1)
        audio_data = torch.from_numpy(int16_to_float32(float32_to_int16(audio_data))).float()
        audio_feature = self.encode_model.get_audio_embedding_from_data(x=audio_data, use_tensor=True)

        text_data = audio_file.split("/")[-1].split("_")[1]
        text_data = [text_data] * 2
        text_feature = self.encode_model.get_text_embedding(text_data, use_tensor=True)
        text_feature = text_feature[0:1, :]
        
        result = 0.5

        # Load human preference
        for key, value in self.data["grouped_ratings"].items():
            for key2, value2 in value.items():
                if audio_file.split("/")[-1] == key2:
                    result = value2
                    break
            else:
                continue
            break

        feature_concat = torch.cat((audio_feature, text_feature), dim=1).to(device) 
        return feature_concat, result

# Define the model
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

AUDIO_FOLDER_PATH = 'Baton/data/Audio/2label_Integrity_HD_Audio'
audio_files = [os.path.join(AUDIO_FOLDER_PATH, f) for f in os.listdir(AUDIO_FOLDER_PATH) if f.endswith('.wav')] #["path_to_audio1.wav", "path_to_audio2.wav", ...]

# CLAP
# quantization
def int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)

def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype(np.int16)

encode_model = laion_clap.CLAP_Module(enable_fusion=False, device=device)
encode_model.load_ckpt()

with open("Baton/data/RM/grouped_ratings_2label.json", "r") as f:
    data = json.load(f)


audio_dataset = AudioDataset(audio_files, encode_model, data)

batch_size = 32
data_loader = DataLoader(audio_dataset, batch_size=batch_size, shuffle=True)

# model = RewardModel()
rewardmodel = RewardModel().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(rewardmodel.parameters(), lr=0.01, weight_decay=1e-5)
EPOCHS = 51

# checkpoint save path
checkpoint_folder = 'rm_ckpt_CLAP_BCE_2label'
os.makedirs(checkpoint_folder, exist_ok=True)

average_losses_per_epoch = []

for epoch in range(EPOCHS):
    losses = 0
    for feature, label in data_loader:
        feature = feature.to(torch.float32).to(device)
        label = label.to(torch.float32).to(device)
        optimizer.zero_grad()
        outputs = rewardmodel(feature)
        loss = criterion(outputs.squeeze(), label)
        losses += loss.item()
        loss.backward(retain_graph=True)
        optimizer.step()
    
    average_loss = losses / len(data_loader)
    if epoch % 10 == 0:
        checkpoint = {
            'epoch': epoch,
            'rewardmodel_state_dict': rewardmodel.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': average_loss,
        }
        # save
        checkpoint_path = os.path.join(checkpoint_folder, f'model_weights_epoch{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
    
    print(f'Epoch {epoch + 1}, Average Loss: {average_loss}')
    average_losses_per_epoch.append(average_loss)

    
plt.plot(average_losses_per_epoch)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("loss")
plt.savefig("loss_CLAP_BCE_2label.png")
plt.show()
