# --------------- 命名统一 ---------------
import json
import glob
import os

with open("/home/hhn/bat/data/Audiocaps/0test_audiocaps_2label.json", 'r') as f:
    data = json.load(f)

folder_path = "/home/hhn/bat/src/eval/output/.../rank_1" # 统一命名前的文件夹(output_0_text.wav)
wav_files = glob.glob(os.path.join(folder_path, '*.wav'))
file_names = [os.path.basename(file) for file in wav_files]

file_path = "/home/hhn/bat/src/eval/output/.../rank_1" # 统一命名后的文件夹(text.wav)

# --------------- tango --------------- 
for file_name in file_names:
    for item in data:
        if file_name.split("_")[2].replace(".wav", "") == item["captions"]:
            new_file_name = item["location"].split("/")[-1]
            origin_file_path = os.path.join(folder_path, file_name)
            after_file_path = os.path.join(file_path, new_file_name)
            os.rename(origin_file_path, after_file_path)

# # --------------- audioldm --------------- 
# for file_name in file_names:
#     for item in data:
#         if file_name.replace(".wav", "") == item["captions"]:
#             new_file_name = item["location"].split("/")[-1]
#             origin_file_path = os.path.join(folder_path, file_name)
#             after_file_path = os.path.join(file_path, new_file_name)
#             os.rename(origin_file_path, after_file_path)

# --------------- 计算指标 ---------------   
import torch
from audioldm_eval import EvaluationHelper

device = torch.device(f"cuda:{0}")

target_audio_path = "/home/hhn/bat/data/Audiocaps/test_2label_16khz"
genration_result_path = "/home/hhn/bat/src/eval/output/.../rank_1"

evaluater = EvaluationHelper(16000, device)
metrics = evaluater.main(
    genration_result_path,
    target_audio_path,
    limit_num=None
)

# # --------------- 误操作后的命名恢复 ---------------
# import json
# import glob
# import os

# with open("/home/hhn/bat/data/Audiocaps/0test_audiocaps_2label.json", "r") as f:
#     data = json.load(f)
    
# folder_path = "/home/hhn/bat/src/eval/output/audioldm_l_2label0.5_epoch20"
# wav_files= glob.glob(os.path.join(folder_path, "*.wav"))
# file_names = [os.path.basename(file) for file in wav_files]

# file_path = "/home/hhn/bat/src/eval/output/audioldm_l_2label0.5_epoch20"

# for file_name in file_names:
#     for item in data:
#         if file_name == item["location"].split('/')[-1]:
#             caption = item["captions"]
#             file_name_target = os.path.join(caption + ".wav")
#             origin_file_path = os.path.join(file_path, file_name)
#             target_file_path = os.path.join(file_path, file_name_target)
#             os.rename(origin_file_path, target_file_path)
            