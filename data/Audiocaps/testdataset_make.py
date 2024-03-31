import json
import os
import re

# # -------------2label test set-------------
# with open("AC_2label.json") as f:
#     AC_2label = json.load(f)
    
# with open("test_audiocaps_subset.json") as f:
#     AC_test = json.load(f)
    
# result = []
# for item1 in AC_2label:
#     for item2 in AC_test:
#         if item1 == item2["captions"]:
#             result.append(item2)
            
# with open("test_audiocaps_2label.json", "w") as f:
#     json.dump(result, f)

# -------------3label test set-------------
with open("AC_3label.json") as f:
    AC_3label = json.load(f)
    
with open("test_audiocaps_subset.json") as f:
    AC_test = json.load(f)
    
result = []
for item1 in AC_3label:
    for item2 in AC_test:
        if item1 == item2["captions"]:
            result.append(item2)
            
with open("test_audiocaps_3label.json", "w") as f:
    json.dump(result, f)