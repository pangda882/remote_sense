export PYTHONPATH=/gemini/code:$PYTHONPATH
export PYTHONPATH="/gemini/code/mamba_old/MambaCD:$PYTHONPATH"

python script/train_MambaBCD.py  --dataset 'SYSU' \
    --batch_size 16 \
    --crop_size 256 \
    --max_iters 320000 \
    --model_type MambaBCD_Base \
    --model_param_path '/gemini/code/MambaCD/changedetection/saved_models' \
    --train_dataset_path '/gemini/data-1/SYSU/train' \
    --train_data_list_path '/gemini/data-1/SYSU/train_list.txt' \
    --test_dataset_path '/gemini/data-1/SYSU/test' \
    --test_data_list_path '/gemini/data-1/SYSU/test_list.txt' \
    --cfg '/gemini/code/MambaCD/changedetection/configs/vssm1/vssm_base_224.yaml' \
    --pretrained_weight_path '/gemini/code/mamba_new/pretrained_weight/vssm_base_0229_ckpt_epoch_237.pth'