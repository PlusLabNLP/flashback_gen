task="story_generation"
lrs=(1e-4)
batch=(64)
seed=5
device="0"
event_num=0
model="facebook/bart-base"
#suffix="gen_no_struct_pipeline_story_input_wp_max500"
#suffix="gen_with_rel_pipeline_story_input_wp_max500"
#suffix="gen_with_rel_pipeline_story_input_wp_max500_rl"

root="../output/${task}/"

for l in "${lrs[@]}"
do
  for s in "${batch[@]}"
  do
      python eval_story_generation_pipeline.py \
      --data_dir "../data/" \
      --model ${model} \
      --topk_sample \
      --task_name  ${task} \
      --gen_storyline_len 512 \
      --file_suffix "_story_generation_wp_baseline_final.json" \
      --device_num ${device} \
      --eval_batch_size 2 \
      --num_train_epochs 10 \
      --learning_rate ${l} \
      --input_event_num ${event_num} \
      --seed ${seed} \
      --model_dir "${root}/${model}_batch_${s}_lr_${l}_seed_${seed}_event_${event_num}_${suffix}/"
  done
done
