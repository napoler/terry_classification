# 运行训练
python3 run_classifier_new.py --data_dir=data --bert_model=bert-base-chinese --task_name=terry --output_dir=run --do_train --no_cuda --num_train_epochs=10

# 运行评估

python3 run_classifier_new.py --data_dir=data --bert_model=bert-base-chinese --task_name=terry --output_dir=run --do_eval --no_cuda --num_train_epochs=10