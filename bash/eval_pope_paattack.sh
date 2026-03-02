#!/bin/bash

baseModel1='LLAVA'
baseModel2='openFlamingo'
baseModel3='LLAVA'

modelPath=openai
modelPath2=veattack_eps2
modelPath2_2=veattack_eps4


# answerFile2="${baseModel1}_${modelPath2}"
# echo "Will save to the following json: "
# echo $answerFile2

# python -m llava.eval.model_vqa_loader_veattack_attention_proto_double \
#     --model-path liuhaotian/llava-v1.5-7b \
#     --eval-model ${baseModel1} \
#     --pretrained_rob_path ${modelPath} \
#     --question-file ./pope_eval/llava_pope_test.jsonl \
#     --image-folder /home/datasets/coco2014/val2014 \
#     --answers-file ./pope_eval/${answerFile2}.jsonl \
#     --temperature 0 \
#     --conv-mode vicuna_v1 \
#     --attack veattack \
#     --eps 2

# # pope_eval/sort_resule.py, sample_jsonl.py, sample_json.py
# python pope_eval/sort_result.py --result-file ./pope_eval/${answerFile2}
# python pope_eval/sample_jsonl.py --result-file ./pope_eval/${answerFile2}
# python pope_eval/sample_json.py --result-file ./pope_eval/${answerFile2}

# python llava/eval/eval_pope_sample.py \
#     --model-name $answerFile2 \
#     --annotation-dir ./pope_eval/coco_sample/ \
#     --question-file ./pope_eval/llava_pope_test_sample.jsonl \
#     --result-file ./pope_eval/${answerFile2}_sorted.jsonl


# answerFile2_2="${baseModel1}_${modelPath2_2}_test"
# echo "Will save to the following json: "
# echo $answerFile2_2

# python -m llava.eval.model_vqa_loader_veattack_attention_proto_double \
#     --model-path liuhaotian/llava-v1.5-7b \
#     --eval-model ${baseModel1} \
#     --pretrained_rob_path ${modelPath} \
#     --question-file ./pope_eval/llava_pope_test.jsonl \
#     --image-folder /home/datasets/coco2014/val2014 \
#     --answers-file ./pope_eval/${answerFile2_2}.jsonl \
#     --temperature 0 \
#     --conv-mode vicuna_v1 \
#     --attack veattack \
#     --eps 4

# # pope_eval/sort_resule.py, sample_jsonl.py, sample_json.py
# python pope_eval/sort_result.py --result-file ./pope_eval/${answerFile2_2}
# python pope_eval/sample_jsonl.py --result-file ./pope_eval/${answerFile2_2}
# python pope_eval/sample_json.py --result-file ./pope_eval/${answerFile2_2}

# python llava/eval/eval_pope_sample.py \
#     --model-name $answerFile2_2 \
#     --annotation-dir ./pope_eval/coco_sample/ \
#     --question-file ./pope_eval/llava_pope_test_sample.jsonl \
#     --result-file ./pope_eval/${answerFile2_2}_sorted.jsonl



# answerFile3="${baseModel2}_${modelPath2}"
# echo "Will save to the following json: "
# echo $answerFile3

# python -m llava.eval.model_vqa_loader_veattack_attention_proto_double \
#     --model-path liuhaotian/llava-v1.5-7b \
#     --eval-model ${baseModel2} \
#     --pretrained_rob_path ${modelPath} \
#     --question-file ./pope_eval/llava_pope_test.jsonl \
#     --image-folder /home/datasets/coco2014/val2014 \
#     --answers-file ./pope_eval/${answerFile3}.jsonl \
#     --temperature 0 \
#     --conv-mode vicuna_v1 \
#     --attack veattack \
#     --eps 2

# # pope_eval/sort_resule.py, sample_jsonl.py, sample_json.py
# python pope_eval/sort_result.py --result-file ./pope_eval/${answerFile3}
# python pope_eval/sample_jsonl.py --result-file ./pope_eval/${answerFile3}
# python pope_eval/sample_json.py --result-file ./pope_eval/${answerFile3}

# python llava/eval/eval_pope_sample.py \
#     --model-name $answerFile3 \
#     --annotation-dir ./pope_eval/coco_sample/ \
#     --question-file ./pope_eval/llava_pope_test_sample.jsonl \
#     --result-file ./pope_eval/${answerFile3}_sorted.jsonl


answerFile3_2="${baseModel2}_${modelPath2_2}"
echo "Will save to the following json: "
echo $answerFile3_2

python -m llava.eval.model_vqa_loader_veattack_attention_proto_double \
    --model-path liuhaotian/llava-v1.5-7b \
    --eval-model ${baseModel2} \
    --pretrained_rob_path ${modelPath} \
    --question-file ./pope_eval/llava_pope_test.jsonl \
    --image-folder /home/datasets/coco2014/val2014 \
    --answers-file ./pope_eval/${answerFile3_2}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --attack veattack \
    --eps 4

# pope_eval/sort_resule.py, sample_jsonl.py, sample_json.py
python pope_eval/sort_result.py --result-file ./pope_eval/${answerFile3_2}
python pope_eval/sample_jsonl.py --result-file ./pope_eval/${answerFile3_2}
python pope_eval/sample_json.py --result-file ./pope_eval/${answerFile3_2}

python llava/eval/eval_pope_sample.py \
    --model-name $answerFile3_2 \
    --annotation-dir ./pope_eval/coco_sample/ \
    --question-file ./pope_eval/llava_pope_test_sample.jsonl \
    --result-file ./pope_eval/${answerFile3_2}_sorted.jsonl




# answerFile4="${baseModel3}_${modelPath2}_13b"
# echo "Will save to the following json: "
# echo $answerFile4

# python -m llava.eval.model_vqa_loader_veattack_attention_proto_double \
#     --model-path liuhaotian/llava-v1.5-13b \
#     --eval-model ${baseModel3} \
#     --pretrained_rob_path ${modelPath} \
#     --question-file ./pope_eval/llava_pope_test.jsonl \
#     --image-folder /home/datasets/coco2014/val2014 \
#     --answers-file ./pope_eval/${answerFile4}.jsonl \
#     --temperature 0 \
#     --conv-mode vicuna_v1 \
#     --attack veattack \
#     --eps 2

# # pope_eval/sort_resule.py, sample_jsonl.py, sample_json.py
# python pope_eval/sort_result.py --result-file ./pope_eval/${answerFile4}
# python pope_eval/sample_jsonl.py --result-file ./pope_eval/${answerFile4}
# python pope_eval/sample_json.py --result-file ./pope_eval/${answerFile4}

# python llava/eval/eval_pope_sample.py \
#     --model-name $answerFile4 \
#     --annotation-dir ./pope_eval/coco_sample/ \
#     --question-file ./pope_eval/llava_pope_test_sample.jsonl \
#     --result-file ./pope_eval/${answerFile4}_sorted.jsonl


# answerFile4_2="${baseModel3}_${modelPath2_2}_13b"
# echo "Will save to the following json: "
# echo $answerFile4_2

# python -m llava.eval.model_vqa_loader_veattack_attention_proto_double \
#     --model-path liuhaotian/llava-v1.5-13b \
#     --eval-model ${baseModel3} \
#     --pretrained_rob_path ${modelPath} \
#     --question-file ./pope_eval/llava_pope_test.jsonl \
#     --image-folder /home/datasets/coco2014/val2014 \
#     --answers-file ./pope_eval/${answerFile4_2}.jsonl \
#     --temperature 0 \
#     --conv-mode vicuna_v1 \
#     --attack veattack \
#     --eps 4

# # pope_eval/sort_resule.py, sample_jsonl.py, sample_json.py
# python pope_eval/sort_result.py --result-file ./pope_eval/${answerFile4_2}
# python pope_eval/sample_jsonl.py --result-file ./pope_eval/${answerFile4_2}
# python pope_eval/sample_json.py --result-file ./pope_eval/${answerFile4_2}

# python llava/eval/eval_pope_sample.py \
#     --model-name $answerFile4_2 \
#     --annotation-dir ./pope_eval/coco_sample/ \
#     --question-file ./pope_eval/llava_pope_test_sample.jsonl \
#     --result-file ./pope_eval/${answerFile4_2}_sorted.jsonl