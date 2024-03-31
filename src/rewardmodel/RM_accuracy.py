import json
import torch
import torch.nn.functional as F
import math

# with open("/home/hhn/tango/pre_AC2label_BCE_transformer_64_0.005.json", "r") as f1:
    # predict = json.load(f1)
    
with open("/home/hhn/tango/pre_AC3label_BCE_transformer_64_0.005_noprefer_nodropout.json", "r") as f1:
    predict = json.load(f1)

with open("/home/hhn/tango/human/human_3label_test.json", "r") as f2:
    gt = json.load(f2)

# with open("/home/hhn/tango/human/human_2label_test.json", "r") as f2:
#     gt = json.load(f2)

#  ------------ 准确率计算 ------------
correct = 0
incorrect = 0
incorrectfile = []
for item1 in predict:
    for item2 in gt["ratings"]:
        for key, value in item2.items():
            for key2, value2 in value.items():
                if item1["audio"].split("/")[-1] == key2:
                    # import pdb;pdb.set_trace()
                    differ = abs(item1["pre_score"][0][0][0] - value2)
                    if differ < 0.5:
                        correct += 1
                    else:
                        incorrectfile.append(item1["audio"].split("/")[-1])
                        incorrect += 1
                        
accuracy = correct/(correct+incorrect)
print("correct is: ", correct)
print("incorrect is: ", incorrect)
print("incorrectfile is: ", incorrectfile)
print("accuracy is: ", accuracy)


#  ------------ 使用PyTorch中的交叉熵损失函数计算 ------------
# correct = 0
# incorrect = 0
# incorrectfile = []
# cross_entropy_loss = 0.0
# num = 0.0

# for item1 in predict:
#     for item2 in gt["ratings"]:
#         for key, value in item2.items():
#             for key2, value2 in value.items():
#                 if item1["audio"].split("/")[-1] == key2:
#                     # import pdb;pdb.set_trace()
#                     prediction = item1["pre_score"][0][0]
#                     target = value2
#                     # partial_loss = - (prediction * math.log(target))
#                     partial_loss = - (target * math.log(prediction) + (1 - target) * math.log(1 - prediction))
#                     # 累加到总的交叉熵损失上
#                     cross_entropy_loss += partial_loss
#                     num = num + 1

# cross_entropy_loss /= num
# print(cross_entropy_loss)