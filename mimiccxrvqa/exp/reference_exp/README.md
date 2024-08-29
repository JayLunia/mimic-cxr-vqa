# MIMIC-DINO

## Pre-training Model

```bash
OMP_NUM_THREADS=16 python -m torch.distributed.launch --nproc_per_node=8 main_dino.py \
--arch vit_small \
--data_path /nfs_data_storage/mmehrqg/mimic-cxr-jpg/20230110/re512_3ch_contour_cropped \
--output_dir ./saved_results/pretraining/boosting \
--epochs 300 \
--teacher_temp 0.07 \
--warmup_teacher_temp_epochs 30 \
--norm_last_layer false   
```


## Fine-tuning Model

```bash
WANDB__SERVICE_WAIT=300 MASTER_PORT=29501 CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 finetune.py \
--num_workers 16 \
--batch_size_per_gpu 256 \
--pretrained_weights ./saved_results/pretraining/boosting/mimicdino/pretrain/vitsmall_boosting.pth \
--dataroot ./mimiccxrvqa/dataset \
--output_dir ./saved_results/finetune/mimic-cxr-max-boost/vit_small/full_finetune/epoch100_minsupp10_all_imgcrop_batch1024_82pair_seed1/ \
--cropping_type img_crop \
--feature_type only_global \
--seed 1 \
--wandb \
--data_aug \
--full_finetune
```