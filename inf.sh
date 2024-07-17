wandb login c84485bd75965fadf8c5d73c1f7acb4abca84df2

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES=0

# heegyu/llama-2-ko-7b-chat, MLP-KTLim/llama-3-Korean-Bllossom-8B

python -m run.decoder \
    --model_id dudcjs2779/dialogue-summarization-T5\
    --device 'cuda:0' \
    --model_ckpt_path '/home/nlplab/ssd1/jeong/Korean_DCS_2024/baseline/Korean_DCS_2024/resource/model/dudcjs2779/dialogue-summarization-T5/2024-07-16-03-36/checkpoint-750' \
    --output './바오밤나무_2.json'

# python -m run.test \
#     --model_id beomi/Llama-3-Open-Ko-8B\
#     --device 'cuda:0' \
#     --model_ckpt_path '/home/nlplab/ssd1/jeong/Korean_DCS_2024/baseline/Korean_DCS_2024/resource/model/beomi/Llama-3-Open-Ko-8B/2024-07-15-20-21/checkpoint-500' \
#     --output './오동나무_inference_st.json'

# python -m run.decoder \
#     --model_id dudcjs2779/dialogue-summarization-T5 \
#     --device 'cuda:0' \
#     --model_ckpt_path '/home/nlplab/ssd1/jeong/Korean_DCS_2024/baseline/Korean_DCS_2024/resource/model/dudcjs2779/dialogue-summarization-T5/2024-07-16-02-32/checkpoint-500' \
#     --output './바오밤나무.json'

# python -m run.test \
#     --model_id beomi/Llama-3-Open-Ko-8B\
#     --device 'cuda:0' \
#     --model_ckpt_path '/home/nlplab/ssd1/jeong/Korean_DCS_2024/baseline/Korean_DCS_2024/resource/model/beomi/Llama-3-Open-Ko-8B/2024-07-15-20-21/checkpoint-500' \
#     --output './오동나무_입력변경.json'

# python -m run.decoder \
#     --model_id lcw99/t5-large-korean-text-summary\
#     --device 'cuda:0' \
#     --model_ckpt_path '/home/nlplab/ssd1/jeong/Korean_DCS_2024/baseline/Korean_DCS_2024/resource/model/lcw99/t5-large-korean-text-summary/2024-07-15-19-25/checkpoint-500' \
#     --output './큰밤나무_early_False.json'


# python -m run.decoder \
#     --model_id lcw99/t5-large-korean-text-summary\
#     --device 'cuda:0' \
#     --model_ckpt_path '/home/nlplab/ssd1/jeong/Korean_DCS_2024/baseline/Korean_DCS_2024/resource/model/lcw99/t5-large-korean-text-summary/2024-07-15-19-25/checkpoint-500' \
#     --output './큰밤나무.json'

# python -m run.decoder \
#     --model_id lcw99/t5-base-korean-text-summary\
#     --device 'cuda:0' \
#     --model_ckpt_path '/home/nlplab/ssd1/jeong/Korean_DCS_2024/baseline/Korean_DCS_2024/resource/model/lcw99/t5-base-korean-text-summary/2024-07-15-18-50/checkpoint-500' \
#     --output './밤나무.json'

#python -m run.test \ (data 형태 : speaker : utterance형태)
#    --model_id beomi/Llama-3-Open-Ko-8B\
#    --device 'cuda:0' \
#    --model_ckpt_path '/home/nlplab/ssd1/jeong/Korean_DCS_2024/baseline/Korean_DCS_2024/resource/model/beomi/Llama-3-Open-Ko-8B/2024-07-13-13-05/checkpoint-500' \
#    --output './오동나무_리트.json'

# python -m run.test \
#     --model_id beomi/Llama-3-Open-Ko-8B-Instruct-preview\
#     --device 'cuda:0' \
#     --model_ckpt_path '/home/nlplab/ssd1/jeong/Korean_DCS_2024/baseline/Korean_DCS_2024/resource/model/beomi/Llama-3-Open-Ko-8B-Instruct-preview/2024-07-13-22-09/checkpoint-202' \
#     --output './오동나무_가짜.json'

# python -m run.test \
#     --model_id beomi/Llama-3-Open-Ko-8B\
#     --device 'cuda:0' \
#     --padding_side "right" \
#     --model_ckpt_path '/home/nlplab/ssd1/jeong/Korean_DCS_2024/baseline/Korean_DCS_2024/resource/model/beomi/Llama-3-Open-Ko-8B/train epoch9' \
#     --output './오동나무_오른쪽.json'

# python -m run.test \
#     --model_id beomi/Llama-3-Open-Ko-8B \
#     --device 'cuda:0' \
#     --model_ckpt_path '/home/nlplab/ssd1/jeong/Korean_DCS_2024/baseline/Korean_DCS_2024/resource/model/beomi/Llama-3-Open-Ko-8B/9' \
#     --output './오동나무_껍질유.json'

# python -m run.test \
#     --model_id beomi/Llama-3-Open-Ko-8B \
#     --device 'cuda:0' \
#     --model_ckpt_path '/home/nlplab/ssd1/jeong/Korean_DCS_2024/baseline/Korean_DCS_2024/resource/model/beomi/Llama-3-Open-Ko-8B/2024-07-08-17-29/checkpoint-500' \
#     --output './오동나무.json'

# python -m run.test \
#     --model_id beomi/Solar-Ko-Recovery-11B \
#     --device 'cuda:0' \
#     --model_ckpt_path '/home/nlplab/ssd1/jeong/Korean_DCS_2024/baseline/Korean_DCS_2024/resource/model/beomi/Solar-Ko-Recovery-11B/2024-07-08-23-18/checkpoint-500' \
#     --output './참나무.json'

