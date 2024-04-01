deepspeed --master_port=7707 --include localhost:0,1,2,3,4,5,6,7 tune_multitask_codet5p.py \
    --model_path Salesforce/codet5p-220m \
    --output_dir models/chkp_all-multitask_codet5p-220m-1024/ \
    --overwrite_output_dir \
    --train_file datasets/all-multitask/train.json \
    --test_file datasets/all-multitask/test.json \
    --cache_train_file datasets/all-multitask/cache-codet5p-220m-multitask/train.cache \
    --cache_test_file datasets/all-multitask/cache-codet5p-220m-multitask/test.cache \
    --overwrite_cache \
    --source_domain "strip_pcode" \
    --target_domain "whole_comment,brief_comment,brief_comment_and_name" \
    --max_source_length 1024 \
    --max_target_length 360 \
    --do_train \
    --do_predict \
    --num_train_epochs 1 \
    --save_strategy 'epoch' \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --weight_decay 0.1 \
    --warmup_steps 200 \
    --predict_with_generate \
    --num_beams 5 \
    --logging_steps 50 \
    --logging_first_step \
    --fp16 \
    --deepspeed ds_stage2_config.json \
    --wandb_project all_multitask_codet5p \
    --wandb_mode offline 


# --overwrite_cache \
# --max_train_samples 500000 \
# --max_test_samples 20000 \
# --max_train_samples 2000 \
# --max_valid_samples 100 \
# --max_test_samples 24 \
# --max_finetune_samples 100 \
