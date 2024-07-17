wandb login c84485bd75965fadf8c5d73c1f7acb4abca84df2


export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
# export CUDA_VISIBLE_DEVICES=0

# heegyu/llama-2-ko-7b-chat, MLP-KTLim/llama-3-Korean-Bllossom-8B, beomi/Solar-Ko-Recovery-11B, beomi/Llama-3-Open-Ko-8B

python -m run.train \
   --model_id  chihoonlee10/T3Q-ko-solar-dpo-v7.0 \
   --device 'cuda:0' \
   --epoch 10 \
   --lr 1e-5 \
   --batch_size 1 \
   --wandb_project 'DCS_wandb' \
   --wandb_entity 'DCS_2024' \
   --wandb_run_name "chihoonlee10/T3Q-ko-solar-dpo-v7.0_test"\
   --top_k 100\
   --top_p 0.95\
   --prompt_type 'mode_with_special_tokens_topic_intro' \
   --prompt " 사용자의 대화를 통해 주요 주제에 대해 요약해주세요." \
   --trainer 'True' \
   --gradient_accumulation_steps 10

# python -m run.train \
#    --model_id  rtzr/ko-gemma-2-9b-it \
#    --device 'cuda:0' \
#    --epoch 10 \
#    --lr 1e-5 \
#    --batch_size 1 \
#    --wandb_project 'DCS_wandb' \
#    --wandb_entity 'DCS_2024' \
#    --wandb_run_name "rtzr/ko-gemma-2-9b-it_test"\
#    --top_k 100\
#    --top_p 0.95\
#    --prompt_type 'mode_with_special_tokens_topic_intro' \
#    --prompt " 사용자의 대화를 통해 주요 주제에 대해 요약해주세요." \
#    --trainer 'True' \
#    --gradient_accumulation_steps 10

# python -m run.train \
#    --model_id  saltlux/Ko-Llama3-Luxia-8B \
#    --device 'cuda:0' \
#    --epoch 10 \
#    --lr 1e-5 \
#    --batch_size 1 \
#    --wandb_project 'DCS_wandb' \
#    --wandb_entity 'DCS_2024' \
#    --wandb_run_name "saltlux/Ko-Llama3-Luxia-8B_test"\
#    --top_k 100\
#    --top_p 0.95\
#    --prompt_type 'mode_with_special_tokens_topic_intro' \
#    --prompt "You are a helpful AI assistant. Please summarize the main topics of the user's conversation. 당신은 유능한 AI 어시스턴트입니다. 사용자의 대화를 통해 주요 주제에 대해 요약해주세요." \
#    --trainer 'True' \
#    --gradient_accumulation_steps 10

# python -m run.train \
#    --model_id  maum-ai/Llama-3-MAAL-8B-Instruct-v0.1 \
#    --device 'cuda:0' \
#    --epoch 10 \
#    --lr 1e-5 \
#    --batch_size 1 \
#    --wandb_project 'DCS_wandb' \
#    --wandb_entity 'DCS_2024' \
#    --wandb_run_name "maum-ai/Llama-3-MAAL-8B-Instruct-v0.1_test"\
#    --top_k 100\
#    --top_p 0.95\
#    --prompt_type 'mode_with_special_tokens_topic_intro' \
#    --prompt "너는 마음에이아이의 챗봇 MAAL이다. 고객의 질문에 친절하게 답하여라. 사용자의 대화를 통해 주요 주제에 대해 요약해주세요." \
#    --trainer 'True' \
#    --gradient_accumulation_steps 10

# python -m run.train \
#    --model_id  spow12/Ko-Qwen2-7B-Instruct \
#    --device 'cuda:0' \
#    --epoch 10 \
#    --lr 1e-5 \
#    --batch_size 1 \
#    --wandb_project 'DCS_wandb' \
#    --wandb_entity 'DCS_2024' \
#    --wandb_run_name "spow12/Ko-Qwen2-7B-Instruct_test"\
#    --top_k 100\
#    --top_p 0.95\
#    --prompt_type 'mode_with_special_tokens_topic_intro' \
#    --prompt "당신은 친절한 챗봇으로서 상대방의 요청에 최대한 자세하고 친절하게 답해야합니다. 사용자가 제공하는 정보를 세심하게 분석하여 사용자의 의도를 신속하게 파악하고 그에 따라 답변을 생성해야합니다. 사용자의 대화를 통해 주요 주제에 대해 요약해주세요." \
#    --trainer 'True' \
#    --gradient_accumulation_steps 10

#python -m run.ed_model_train \
#   --model_id  dudcjs2779/dialogue-summarization-T5 \
#   --device 'cuda:0' \
#   --epoch 15 \
#   --lr 1e-5 \
#   --batch_size 1 \
#   --wandb_project 'DCS' \
#   --wandb_entity 'wjdghwns1096' \
#   --wandb_run_name "T5_summarization_dialogue"\
#   --top_k 100\
#   --top_p 0.95\
#   --prompt_type 'mode_with_special_tokens_topic_intro' \
#   --prompt "사용자의 대화를 통해 주요 주제에 대해 요약해주세요." \
#   --trainer 'False' \
#   --gradient_accumulation_steps 10


# python -m run.encoder \
#    --model_id  dudcjs2779/dialogue-summarization-T5 \
#    --device 'cuda:0' \
#    --epoch 10 \
#    --lr 2e-5 \
#    --batch_size 1 \
#    --wandb_project 'DCS' \
#    --wandb_entity 'wjdghwns1096' \
#    --prompt_type 'mode_with_special_tokens_topic_intro' \
#    --prompt "사용자의 대화를 통해 주요 주제에 대해 요약해주세요." \
#    --trainer 'True' \
#    --gradient_accumulation_steps 10

# python -m run.train \
#    --model_id  beomi/Llama-3-Open-Ko-8B \ (데이터 형태 변환(speaker : utterance -> speaker가 utterance라고 말했다))
#    --device 'cuda:0' \
#    --epoch 10 \
#    --lr 2e-5 \
#    --batch_size 1 \
#    --wandb_project 'DCS' \
#    --wandb_entity 'wjdghwns1096' \
#    --prompt_type 'mode_with_special_tokens_topic_intro' \
#    --prompt "You are a helpful AI assistant. Please summarize the main topics of the user's conversation. 당신은 유능한 AI 어시스턴트입니다. 사용자의 대화를 통해 주요 주제에 대해 요약해주세요." \
#    --trainer 'True' \
#    --gradient_accumulation_steps 10

# python -m run.encoder \
#    --model_id  lcw99/t5-large-korean-text-summary \
#    --device 'cuda:0' \
#    --epoch 10 \
#    --lr 2e-5 \
#    --batch_size 1 \
#    --wandb_project 'DCS' \
#    --wandb_entity 'wjdghwns1096' \
#    --prompt_type 'mode_with_special_tokens_topic_intro' \
#    --prompt "You are a helpful AI assistant. Please summarize the main topics of the user's conversation. 당신은 유능한 AI 어시스턴트입니다. 사용자의 대화를 통해 주요 주제에 대해 요약해주세요." \
#    --trainer 'True' \
#    --gradient_accumulation_steps 10

# python -m run.encoder \
#    --model_id  lcw99/t5-base-korean-text-summary \
#    --device 'cuda:0' \
#    --epoch 10 \
#    --lr 2e-5 \
#    --batch_size 1 \
#    --wandb_project 'DCS' \
#    --wandb_entity 'wjdghwns1096' \
#    --prompt_type 'mode_with_special_tokens_topic_intro' \
#    --prompt "You are a helpful AI assistant. Please summarize the main topics of the user's conversation. 당신은 유능한 AI 어시스턴트입니다. 사용자의 대화를 통해 주요 주제에 대해 요약해주세요." \
#    --trainer 'True' \
#    --gradient_accumulation_steps 10

#python -m run.encoder-decoder \
#   --model_id  paust/pko-t5-base \
#   --device 'cuda:0' \
#   --epoch 10 \
#   --lr 2e-5 \
#   --batch_size 1 \
#   --wandb_project 'DCS' \
#   --wandb_entity 'wjdghwns1096' \
#   --prompt_type 'mode_with_special_tokens_topic_intro' \
#   --prompt "You are a helpful AI assistant. Please summarize the main topics of the user's conversation. 당신은 유능한 AI 어시스턴트입니다. 사용자의 대화를 통해 주요 주제에 대해 요약해주세요." \
#   --trainer 'True' \
#   --gradient_accumulation_steps 10

# python -m run.train \
#    --model_id  beomi/Llama-3-Open-Ko-8B-Instruct-preview\
#    --device 'cuda:0' \
#    --epoch 10 \
#    --lr 2e-5 \
#    --batch_size 1\
#    --wandb_project 'DCS' \
#    --wandb_entity 'wjdghwns1096' \
#    --prompt_type 'mode_with_special_tokens_topic_intro' \
#    --prompt "You are a helpful AI assistant. Please summarize the main topics of the user's conversation. 당신은 유능한 AI 어시스턴트입니다. 사용자의 대화를 통해 주요 주제에 대해 요약해주세요." \
#    --trainer 'True' \
#    --gradient_accumulation_steps 10


# python -m run.train \
#    --model_id  beomi/Llama-3-Open-Ko-8B-Instruct-preview\
#    --device 'cuda:0' \
#    --epoch 10 \
#    --lr 2e-5 \
#    --batch_size 1\
#    --wandb_project 'DCS' \
#    --wandb_entity 'wjdghwns1096' \
#    --prompt_type 'mode_with_special_tokens_topic_intro' \
#    --prompt "You are a helpful AI assistant. Please summarize the main topics of the user's conversation. 당신은 유능한 AI 어시스턴트입니다. 사용자의 대화를 통해 주요 주제에 대해 요약해주세요." \
#    --trainer 'True' \
#    --gradient_accumulation_steps 10

# python -m run.train \
#    --model_id  beomi/Llama-3-Open-Ko-8B\
#    --device 'cuda:0' \
#    --epoch 10 \
#    --lr 2e-5 \
#    --batch_size 1\
#    --wandb_project 'DCS' \
#    --wandb_entity 'wjdghwns1096' \
#    --prompt_type 'mode_with_special_tokens_topic_intro' \
#    --prompt "You are a helpful AI assistant. Please summarize the main topics of the user's conversation. 당신은 유능한 AI 어시스턴트입니다. 사용자의 대화를 통해 주요 주제에 대해 요약해주세요." \
#    --trainer 'True' \
#    --gradient_accumulation_steps 10

# python -m run.train \
#    --model_id  beomi/Llama-3-Open-Ko-8B\
#    --device 'cuda:0' \
#    --epoch 10 \
#    --lr 1e-5 \
#    --batch_size 1\
#    --padding_side "right" \
#    --wandb_project 'DCS' \
#    --wandb_entity 'wjdghwns1096' \
#    --trainer 'False' \
#    --gradient_accumulation_steps 10

# python -m run.train \
#    --model_id  nlpai-lab/KULLM3\
#    --device 'cuda:0' \
#    --epoch 10 \
#    --lr 1e-5 \
#    --batch_size 1\
#    --wandb_project 'DCS' \
#    --wandb_entity 'wjdghwns1096' \
#    --trainer 'False' \
#    --gradient_accumulation_steps 10


# python -m run.train \
#    --model_id beomi/Llama-3-Open-Ko-8B \
#    --device 'cuda:0' \
#    --epoch 10 \
#    --lr 2e-5 \
#    --batch_size 2\
#    --wandb_project 'DCS' \
#    --wandb_entity 'wjdghwns1096' \
#    --trainer 'False' \
#    --gradient_accumulation_steps 10

# python -m run.train \
#    --model_id beomi/Solar-Ko-Recovery-11B \
#    --device 'cuda:0' \
#    --epoch 10 \
#    --lr 2e-5 \
#    --wandb_project 'DCS' \
#    --wandb_entity 'wjdghwns1096' \
#    --trainer 'True' \
#    --gradient_accumulation_steps 10

# python -m run.train \ 2024-07-08-20-13
#    --model_id beomi/Solar-Ko-Recovery-11B \
#    --device 'cuda:0' \
#    --epoch 10 \
#    --lr 2e-5 \
#    --wandb_project 'DCS' \
#    --wandb_entity 'wjdghwns1096' \
#    --trainer 'False' \
#    --gradient_accumulation_steps 10

# python -m run.train \2024-07-08-17-28
#     --model_id beomi/Llama-3-Open-Ko-8B \
#     --device 'cuda:0' \
#     --epoch 10 \
#     --lr 1e-5 \
#     --wandb_project 'DCS' \
#     --wandb_entity 'wjdghwns1096' \
#     --trainer 'True' \
#     --gradient_accumulation_steps 10

# python -m run.train \
#     --model_id MLP-KTLim/llama-3-Korean-Bllossom-8B \
#     --device 'cuda:0' \
#     --epoch 10 \
#     --lr 2e-5 \
#     --wandb_project 'Korean_DCS_2024' \
#     --wandb_entity 'leoyeon' \
#     --trainer 'True' \
#     --gradient_accumulation_steps 10

