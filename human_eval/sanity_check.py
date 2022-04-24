import json
with open("wp_human_evaluation.json") as infile:
    to_check = json.load(infile)

print(len(to_check))

model_names = ['baseline', 'vanilla', 'rl', 'structured_prompt']

filenames = ['stories_from_content_planning_baseline.json',
             'stories_from_gen_no_struct_pipeline_story_input_wp_max500_topk_sample_5.json',
             'stories_from_gen_with_rel_pipeline_story_input_wp_max500_rl_topk_sample_5.json',
             'stories_from_gen_with_rel_pipeline_story_input_wp_max500_topk_sample_5.json']

data = []
for f in filenames:
    with open("../generation_wp/%s" % f) as infile:
        data.append(json.load(infile))


for i, text, m in zip(to_check['passage_id'], to_check['generated_text'], to_check['model']):
    idx = model_names.index(m)

    if data[idx][i] != text:
        print(m)
        print(text)
        print(">>>")
        print(data[idx][i])
        print("==========")


# def jaccard_similarity(list1, list2):
#     intersection = len(list(set(list1).intersection(list2)))
#     union = (len(list1) + len(list2)) - intersection
#     return float(intersection) / union
#
# with open("roc_human_evaluation.json") as infile:
#     to_check = json.load(infile)
#
# print(len(to_check))
#
# model_names = ['baseline', 'vanilla', 'structured_prompt', 'pretrained', 'rl']
#
# filenames = ['stories_from_megatron_124m.json',
#              'stories_from_gen_with_no_struct_pipeline_story_input_5.json',
#              'stories_from_gen_with_rel_output_pipeline_story_input_5.json',
#              'stories_from_gen_with_rel_output_pipeline_pretrain_story_input_1M_5.json',
#              'stories_from_gen_with_rel_output_pretrain_pipeline_story_input_rl_5.json']
#
# data = []
# for f in filenames:
#     with open("../generation/%s" % f) as infile:
#         temp = json.load(infile)
#         data.append(temp)
#         print(len(temp))
#
# count = 0
# for i, text, m in zip(to_check['passage_id'], to_check['generated_text'], to_check['model']):
#     idx = model_names.index(m)
#
#     if m == 'baseline':
#         a = text.split(' ')
#         for j, ex in enumerate(data[0]):
#             b = ex.split(' ')
#             if jaccard_similarity(a, b) > 0.3:
#                 print(i, j)
#                 print(a)
#                 print(">>>")
#                 print(b)
#                 print("==========\n")
#                 count += 1
# print(count)
        # print(m)
        # print(text)
        #
        # print(data[idx][int(i)])


