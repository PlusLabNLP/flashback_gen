task="story_generation"
lrs=(5e-5)
batch=(10)
seeds=(5)
device="0"
model="facebook/bart-base"
event_num=0
root="../output/"
suffix="gen_with_rel_output_pretrain_pipeline_story_input_rl"

for l in "${lrs[@]}"
do
  for s in "${batch[@]}"
  do
	    for seed in "${seeds[@]}"
	    do
	        load_model_dir="${root}${task}/${model}_batch_32_lr_5e-5_seed_5_event_0_pretrain_with_rel_5sents_output_1M/pytorch_model.bin"
          nohup python run_story_generation_pipeline_rl.py \
          --load_model ${load_model_dir} \
          --data_dir "../data/" \
          --model ${model} \
          --save_model \
          --task_name  ${task} \
          --file_suffix "_story_generation_all_complete.json" \
          --device_num ${device} \
          --train_batch_size ${s} \
          --eval_batch_size ${s} \
          --num_train_epochs 10 \
          --max_seq_length 72 \
          --do_train \
          --do_eval \
          --input_event_num ${event_num} \
          --learning_rate ${l} \
          --seed ${seed} \
          --output_dir "${root}${task}/${model}_batch_${s}_lr_${l}_seed_${seed}_event_${event_num}_${suffix}" \
          > ./logs/${task}/${model}_batch_${s}_lr_${l}_seed_${seed}_event_${event_num}_${suffix}
      done
    done
done