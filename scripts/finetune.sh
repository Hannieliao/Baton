accelerate launch --main_process_port 29052 src/t2amodel/tango/finetune.py \
--finetune_file="data/finetune_HARA.json \
--train_file="data/finetune_ACpretrain_annotation.json" \
--validation_file="data/valid_audiocaps.json" \
--test_file="data/test_audiocaps_subset.json" \
--beta 0.25 --text_encoder_name "ckpt/google-flan-t5-large" --scheduler_name "ckpt/stable-diffusion-2-1/scheduler_config.json" --unet_model_config "configs/diffusion_model_config.json" \
--hf_model "tango/ckpt/tango_full_ft" --freeze_text_encoder \
--gradient_accumulation_steps 4 --per_device_train_batch_size=6 --per_device_eval_batch_size=6 --augment \
--learning_rate=1e-5 --num_train_epochs 10 --snr_gamma 5.0 \
--text_column=captions --audio_column=location --feedback_column=feedback --checkpointing_steps="best"