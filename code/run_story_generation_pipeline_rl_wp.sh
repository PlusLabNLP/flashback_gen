task="story_generation"
lrs=(1e-4)
batch=(64)
seeds=(5)
device="3"
model="facebook/bart-base"
event_num=0
root="../output/"
suffix="gen_with_rel_pipeline_story_input_wp_max500_rl"
for l in "${lrs[@]}"
do
  for s in "${batch[@]}"
  do
	    for seed in "${seeds[@]}"
	    do
          nohup python run_story_generation_pipeline_rl.py \
          --data_dir "../data/" \
          --model ${model} \
          --gradient_accumulation_steps 8 \
          --save_model \
          --task_name  ${task} \
          --gen_storyline_len 512 \
          --file_suffix "_story_generation_wp_max500.json" \
          --device_num ${device} \
          --train_batch_size ${s} \
          --eval_batch_size ${s} \
          --num_train_epochs 10 \
          --do_train \
          --do_eval \
          --input_event_num ${event_num} \
          --learning_rate ${l} \
          --seed ${seed} \
          --output_dir "${root}/${task}/${model}_batch_${s}_lr_${l}_seed_${seed}_event_${event_num}_${suffix}" \
          > ./logs/${task}/${model}_batch_${s}_lr_${l}_seed_${seed}_event_${event_num}_${suffix}
      done
    done
done