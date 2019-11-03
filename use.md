# 运行训练 评估

python run_glue.py \
  --model_type bert \
  --model_name_or_path bert-base-chinese \
  --task_name terry \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir data/ \
  --max_seq_length 32 \
  --per_gpu_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir data/terry_output/




# 运行评估

