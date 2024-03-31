# Inference after fine_tuning with human feedback
# 2label_Integrity
CUDA_VISIBLE_DEVICES=0 python bat/src/t2amodel/tango/inference_multilabel.py \
--original_args="bat/src/t2amodel/tango/saved/1681728144/summary.jsonl" \
--test_file="bat/data/Audiocaps/test_audiocaps_2label.json" \
--test_references="bat/data/Audiocaps/test_2label" \
--model="bat/src/t2amodel/tango/saved/1681728144/epoch_10/pytorch_model_2.bin" \
--num_steps 200 --guidance 3 --num_samples 1
# 3label_Temporal
# CUDA_VISIBLE_DEVICES=0 python bat/src/t2amodel/tango/inference_multilabel.py \
# --original_args="bat/src/t2amodel/tango/saved/1681728144/summary.jsonl" \
# --test_file="bat/data/Audiocaps/test_audiocaps_3label.json" \
# --test_references="bat/data/Audiocaps/test_3label" \
# --model="bat/src/t2amodel/tango/saved/1681728144/epoch_10/pytorch_model_2.bin"\
# --num_steps 200 --guidance 3 --num_samples 1