wandb login 20f894088a42a42e5eef02b48b1e6cce6805fdfe

# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1
# CUDA_VISIBLE_DEVICES=0,1

# heegyu/llama-2-ko-7b-chat, MLP-KTLim/llama-3-Korean-Bllossom-8B, beomi/Solar-Ko-Recovery-11B, beomi/Llama-3-Open-Ko-8B

python -m run.train \
    --model_id beomi/Solar-Ko-Recovery-11B \
    --device 'cuda:0' \
    --epoch 10 \
    --lr 2e-5 \
    --wandb_project 'Korean_DCS_2024' \
    --wandb_entity 'leoyeon' \
    --trainer 'True' \
    --gradient_accumulation_steps 10

# python -m run.train \
#     --model_id beomi/Llama-3-Open-Ko-8B \
#     --device 'cuda:0' \
#     --epoch 10 \
#     --lr 2e-5 \
#     --wandb_project 'Korean_DCS_2024' \
#     --wandb_entity 'leoyeon' \
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

