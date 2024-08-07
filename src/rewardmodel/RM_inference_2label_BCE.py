import os
import json
import torch
import numpy as np
import librosa
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import laion_clap

def int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)

def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype(np.int16)

class AudioDataset:
    def __init__(self, audio_files, encode_model):
        self.audio_files = audio_files
        self.encode_model = encode_model

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_file = self.audio_files[idx]

        audio_data, _ = librosa.load(audio_file, sr=16000)
        audio_data = audio_data.reshape(1, -1)
        audio_data = torch.from_numpy(int16_to_float32(float32_to_int16(audio_data))).float()
        audio_feature = self.encode_model.get_audio_embedding_from_data(x=audio_data, use_tensor=True)

        text_data = audio_file.split("/")[-1].split("_")[-1].replace(".wav", "")
        text_data = [text_data] * 2
        text_feature = self.encode_model.get_text_embedding(text_data, use_tensor=True)
        text_feature = text_feature[0:1, :]

        feature_concat = torch.cat((audio_feature, text_feature), dim=1)
        return feature_concat, audio_file

class RewardModel(torch.nn.Module):
    def __init__(self):
        super(RewardModel, self).__init__()
        self.fc1 = torch.nn.Linear(1024, 512)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(512, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

AUDIO_FOLDER_PATH = 'Baton/data/Audio/2label_Integrity_RD_Audio'
audio_files = [os.path.join(AUDIO_FOLDER_PATH, f) for f in os.listdir(AUDIO_FOLDER_PATH) if f.endswith('.wav')]

encode_model = laion_clap.CLAP_Module(enable_fusion=False, device='cuda:0')
encode_model.load_ckpt()

audio_dataset = AudioDataset(audio_files, encode_model)
data_loader = DataLoader(audio_dataset, batch_size=16, shuffle=False)

reward_model = RewardModel().to('cuda:0')
checkpoint = torch.load('rm_ckpt_CLAP_BCE_2label/model_weights_epoch50.pth') # Replace your checkpoint path
reward_model.load_state_dict(checkpoint['rewardmodel_state_dict'])
reward_model.eval()

results = []
with torch.no_grad():
    for batch_idx, (feature, audio_file) in enumerate(data_loader):
        feature = feature.to(torch.float32).to('cuda:0')
        outputs = reward_model(feature)
        predictions = outputs.cpu().numpy().tolist()
        for i in range(len(audio_file)):
            result = {"audio": audio_file[i], "pre_score": predictions[i][0]}
            results.append(result)

output_json_path = '2label_RA.json'
with open(output_json_path, 'w') as json_file:
    json.dump(results, json_file, indent=4)
