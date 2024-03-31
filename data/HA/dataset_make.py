import json
import os
import re

audios_path = '/home/hhn/bat/data/Audio/2label_Integrity_HD_Audio'
files = os.listdir(audios_path)
audios_dir = [os.path.join(audios_path, file) for file in files]

dataset = []
# -------------load human annotaiton-------------
with open("HA_2label_Integrity.json") as f:
    humanscore = json.load(f)

for audio_dir in audios_dir:
    audioname = audio_dir.split('/')[-1]
    for key, value in humanscore.items():
        for key1, value1 in value.items():
            for key2, value2 in value1.items():
                if audioname == key2:
                    data = {
                        "dataset": "2label_HA",
                        "location": audio_dir,
                        "captions": audioname.split('_')[1],
                        "feedback": value2
                    }
                    dataset.append(data)
            
with open("2label_HA.json", "w") as f:
    json.dump(dataset, f)

# -------------load RM annotaiton-------------
# with open("") as f:
#     rmscore = json.load(f)

# for audio_dir in audios_dir:
#     audioname = audio_dir.split('/')[-1]
#     for item in rmscore:
#         if audioname == item['audio'].split('/')[-1]:
#             data = {
#                 "dataset": "2label_rm",
#                 "location": audio_dir,
#                 "captions": audioname,
#                 "feedback": item["score"]
#             }
#             dataset.append(data)
            
# with open("rm_predict_2label.json", "w") as f:
#     json.dump(dataset, f)
    