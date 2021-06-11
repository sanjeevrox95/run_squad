
import os
import sys
import traceback
import shutil
import subprocess


training_dir ='E:/Albert_QA'
testing_dir ='E:/Albert_QA'
training_file =os.path.join(training_dir,'train_data_30.json')
testing_file = os.path.join(testing_dir,'dev_v2.json')
subprocess.run('python run_squad.py \
--model_type albert \
--model_name_or_path ktrapeznikov/albert-xlarge-v2-squad-v2 \
--do_train \
--do_eval \
--do_lower_case \
--train_file ' + training_file + ' \
--predict_file ' + testing_file + ' \
--per_gpu_train_batch_size 3 \
--learning_rate 3e-5 \
--num_train_epochs 1.0 \
--max_seq_length 384 \
--doc_stride 128 \
--output_dir ../model \
--save_steps 1000 \
--threads 4 \
--version_2_with_negative', shell=True)
print('Training complete!!')

