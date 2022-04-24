# Flashback Generation
This is the public repo for our NAACL:'22 paper: Go Back in Time: Generating Flashbacks in Stories with Event Plots and
Temporal Prompts.

### 0. Preliminary
- environment: for reference, we export our environment in env.yml.
- data: download data from here: https://drive.google.com/drive/folders/1LfHRfE_0HELmnKGOBT_vVB8PIiMArk1X?usp=sharing, and save it into `./data`
- trained models: download trained models from the same link and save them into `./output/story_generation/facebook/`. 


### 1. Replicating Results
##### 1.1 ROCStories
- `cd code` and `bash eval_story_generation_pipeline.sh`.
- Set `seed` to one of (5, 9998 and 20016) in the above bash file, and `suffix` for different models.  
    - Vanilla-Gen: `suffix=gen_with_no_struct_pipeline_story_input`
    - Structured-Prompt: `suffix=gen_with_rel_output_pipeline_story_input`
    - Pretrained: `suffix=gen_with_rel_output_pipeline_pretrain_story_input_1M`
    - RL: `suffix=gen_with_rel_output_pretrain_pipeline_story_input_rl`
- After executing these scripts, generated stories will be saved in `./generation`.
- Then run `python cal_evaluation_metrics.py` to evaluate these stories. If you have different story outputs, make sure to set `--filenames` correctly.
- For human evaluation results and significance test, `cd human_eval` and `python cal_human_eval.py --data_file roc_human_evaluation.json`.
    - Note: you can use `passage_id` in `roc_human_evaluation.json` to retrieve samples in `./generation` except for the baseline `stories_from_megatron_124m.json`, which has different sample order. We use string/token match to resolve the issue.


##### 1.2 WritingPrompts
- `cd code` and `bash eval_story_generation_pipeline_wp.sh`.
- All you need to change is `suffix` in the above bash file.
    - Vanilla-Gen: `suffix=gen_no_struct_pipeline_story_input_wp_max500`
    - Structured-Prompt: `suffix=gen_with_rel_pipeline_story_input_wp_max500`
    - RL: `suffix=gen_with_rel_pipeline_story_input_wp_max500_rl`
- After executing these scripts, generated stories will be saved in `./generation_wp`.
- Then run `python cal_evaluation_metrics_wp.py` to evaluate these stories. Again, if you have different story outputs, make sure to set `--filenames` correctly.
- For human evaluation results and significance test, `cd human_eval` and `python cal_human_eval.py --data_file wp_human_evaluation.json`.
- Note: We decode WP stories using topk-sample option. We found that due to this randomness, different platforms, different pytorch/huggingface versions would result in different decoded stories even after fixing random seeds. We initially decode stories on AWS EC2 instances, but results change by switching to local clusters. But we observed the overall quality of stories hold.


### 2. Model Training
We also provide code to train our models. Training ROC stories is relatively easy, but due to long sentences in WritingPrompts data, a GPU with large memory such as A100 is required to trained efficiently.
##### 1.1 ROCStories
- `cd code`, you are welcome to try other hyper-parameters, but the key is to set `suffix` correctly.
- Vanilla-Gen 
    - set `suffix=gen_with_no_struct_pipeline_story_input`
    - `bash run_story_generation_pipeline.sh` 
- Structured-Prompt
    - set `suffix=gen_with_rel_output_pipeline_story_input`
    - `bash run_story_generation_pipeline.sh` 
- Pretrained
    - set `suffix=gen_with_rel_output_pipeline_pretrain_story_input_1M`
    - add `--load_model ${load_model_dir}` to load pretrained storyline model. You should have downloaded in `./output` and we provide the directory in the script.
    - `bash run_story_generation_pipeline.sh` 
- RL
    - set `suffix=gen_with_rel_output_pretrain_pipeline_story_input_rl`
    - add `--load_model ${load_model_dir}` as above
    - `bash run_story_generation_pipeline_rl.sh`, notice a different script here.
    

##### 1.2 WritingPrompts
- `cd code`, again you are welcome to try other hyper-parameters.
- Vanilla-Gen
    - set `suffix=gen_no_struct_pipeline_story_input_wp_max500`
    - `bash run_story_generation_pipeline_wp.sh` 
- Structured-Prompt
    - set `suffix=gen_with_rel_pipeline_story_input_wp_max500`
    - `bash run_story_generation_pipeline_wp.sh` 
- RL
    - set `suffix=gen_with_rel_pipeline_story_input_wp_max500_rl`
    - `bash run_story_generation_pipeline_rl_wp.sh` 