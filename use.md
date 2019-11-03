# 运行训练
python3 run_classifier_new.py --data_dir=data --bert_model=bert-base-chinese --task_name=terry --output_dir=run --do_train --no_cuda --num_train_epochs=10


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

python3 run_classifier_new.py --data_dir=data --bert_model=bert-base-chinese --task_name=terry --output_dir=run --do_eval --no_cuda --num_train_epochs=10