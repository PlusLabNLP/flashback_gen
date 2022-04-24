task="story_generation"
lrs=(5e-5)
batch=(10)
seed=5 # 9998, 20016
device="0"
event_num=0
model="facebook/bart-base"
suffix="gen_with_no_struct_pipeline_story_input"
#suffix="gen_with_rel_output_pipeline_story_input"
#suffix="gen_with_rel_output_pipeline_pretrain_story_input_1M"
#suffix="gen_with_rel_output_pretrain_pipeline_story_input_rl"

root="../output/${task}/"
for l in "${lrs[@]}"
do
  for s in "${batch[@]}"
  do
      python eval_story_generation_pipeline.py \
      --data_dir "../data/" \
      --model ${model} \
      --task_name  ${task} \
      --file_suffix "_story_generation_all_complete_final.json" \
      --device_num ${device} \
      --eval_batch_size ${s} \
      --num_train_epochs 10 \
      --learning_rate ${l} \
      --input_event_num ${event_num} \
      --seed ${seed} \
      --model_dir "${root}/${model}_batch_${s}_lr_${l}_seed_${seed}_event_${event_num}_${suffix}/"
  done
done
