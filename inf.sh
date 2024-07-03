wandb login 20f894088a42a42e5eef02b48b1e6cce6805fdfe

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
#export CUDA_VISIBLE_DEVICES=0

# heegyu/llama-2-ko-7b-chat, MLP-KTLim/llama-3-Korean-Bllossom-8B
python -m run.test \
    --model_id MLP-KTLim/llama-3-Korean-Bllossom-8B \
    --device 'cuda:0' \
    --model_ckpt_path '/home/nlplab/ssd1/YU/말평/DCS/Korean_DCS_2024/resource/model/checkpoint-500' \
    --output './test.json'
