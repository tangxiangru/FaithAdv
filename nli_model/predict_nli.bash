python run_nli.py \
    --train_file data/nli_training_data/train.json \
    --validation_file data/nli_training_data/dev.json \
    --test_file advfact_data.json \
    --do_predict \
    --max_seq_length 512 \
    --per_device_eval_batch_size 32 \
    --output_dir /root/autodl-tmp/longeval_nli_prediction \
    --model_name_or_path roberta-large-mnli

    # --model_name_or_path /root/autodl-tmp/longeval_nli_model/checkpoint-1000