import json

with open("/home/hhn/bat/src/rewardmodel/human_2label_test.json", "r") as f:
    humanscore = json.load(f)
    
with open("/home/hhn/bat/data/RM/2label_RA.json", "r") as f:
    data = json.load(f)
    
correctness = []
incorrectness = []

for gt in humanscore['ratings']:
    for key, value in gt.items():
        for key2, value2 in value.items():
            if value2 == 1:
                correctness.append(key2)
            else:
                incorrectness.append(key2)

TPs = []
FPs = []
for item in data:
    for correct in correctness:
        if item["audio"].split("/")[-1] == correct:
            TP = {
                "audio": item["audio"].split("/")[-1],
                "predict": item["pre_score"][0]
            }
            TPs.append(TP)
    for incorrect in incorrectness:
        if item["audio"].split("/")[-1] == incorrect:
            FP = {
                "audio": item["audio"].split("/")[-1],
                "predict": item["pre_score"][0]
            }
            FPs.append(FP)

with open("correctness_pre2label.json", "w") as f:
    json.dump(correctness, f)

with open("incorrectness_pre2label.json", "w") as f:
    json.dump(incorrectness, f)

with open("truepredict_pre2label.json", "w") as f:
    json.dump(TP, f)
    
with open("falsepredict_pre2label.json", "w") as f:
    json.dump(FP, f)
    
# 小提琴图绘制
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

with open("correctness_pre2label.json", "r") as f:
    human_1_data = json.load(f)

with open("incorrectness_pre2label.json", "r") as f:
    human_0_data = json.load(f)

with open("truepredict_pre2label.json", "r") as f:
    pred_1_data = json.load(f)
    
with open("falsepredict_pre2label.json", "r") as f:
    pred_0_data = json.load(f)
            
pred_1_value = [item["predict"] for item in pred_1_data]
pred_0_value = [item["predict"] for item in pred_0_data]

df = [pred_1_value, pred_0_value]

plt.violinplot(df, positions=[0,1])

plt.title("Violin")
plt.xlabel("True Label")
plt.ylabel("Predicted Value")

plt.savefig('Violin_2label.png')
plt.sshow()               
        
    
    
