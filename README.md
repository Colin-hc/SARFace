# SARFace: Segmentation-Aware Face Recognition with Reinforcement Learning

Official github repository for SARFace: Segmentation-Aware Face Recognition with Reinforcement Learning 


> Abstract: Face recognition in unconstrained environments presents significant challenges due to complex backgrounds, occlusions, and vari-
ations in facial poses. These factors cause different facial regions to contribute variably to recognition accuracy, with some regions
even negatively affecting the outcomes. Thus, effectively identifying and extracting facial regions that positively impact recogni-
tion performance is critical to improving accuracy. To address this issue, we propose a reinforcement learning-based face recognition
framework that dynamically identifies and prioritizes effective regions in facial images captured in the wild. By leveraging facial se-
mantic information, our method eliminates regions that hinder recognition, thereby enhancing overall performance. The framework
employs a three-network architecture: the first network segments the facial image to extract semantic information, the second net-
work utilizes this information to identify and prioritize key regions for recognition, and the third network extracts discriminative fea-
tures from these regions. Reinforcement learning is leveraged to train a Policy Network, which determines the optimal partitioning
strategy for complex face recognition tasks. 

`
# Pretrain Script

```
python main.py \
    --data_root  \
    --train_data_path faces_vgg_112x112 \
    --val_data_path faces_emore \
    --prefix ir50_vgg_adaface \
    --use_wandb \
    --use_mxrecord \
    --gpus 4 \
    --use_16bit \
    --arch ir_50 \
    --batch_size 128 \
    --num_workers 4 \
    --epochs 26 \
    --lr_milestones 12,20,24 \
    --lr 0.0001 \
    --head cosface \
    --m 0.4 \
    --h 0.333 \
    --low_res_augmentation_prob 0\
    --crop_augmentation_prob 0\
    --photometric_augmentation_prob 0
```
# Train Script
```
python main_sarface.py \
    --data_root \
    --train_data_path faces_vgg_112x112 \
    --val_data_path faces_emore \
    --prefix ir50_vgg_adaface \
    --use_wandb \
    --use_mxrecord \
    --gpus 4 \
    --use_16bit \
    --arch ir_50 \
    --batch_size 128 \
    --num_workers 16 \
    --epochs 26 \
    --lr_milestones 12,20,24 \
    --lr 0.1 \
    --head adaface \
    --m 0.4 \
    --h 0.333 \
    --low_res_augmentation_prob 0.2 \
    --crop_augmentation_prob 0.2 \
    --photometric_augmentation_prob 0.2
```
# Evaluation Script
```
python main_sarface.py \
    --data_root \
    --train_data_path faces_vgg_112x112 \
    --val_data_path faces_emore \
    --prefix ir50_vgg_adaface \
    --use_wandb \
    --use_mxrecord \
    --gpus 4 \
    --use_16bit \
    --arch ir_50 \
    --batch_size 128 \
    --num_workers 16 \
    --epochs 26 \
    --lr_milestones 12,20,24 \
    --lr 0.1 \
    --head adaface \
    --m 0.4 \
    --h 0.333 \
    --evaluation\
    --low_res_augmentation_prob 0.2 \
    --crop_augmentation_prob 0.2 \
    --photometric_augmentation_prob 0.2
```

# Preapring Dataset and Training Scripts
- Please refer to [README_TRAIN.md](./README_TRAIN.md)
- [IMPORTANT] Note that our implementation assumes that input to the model is `BGR` color channel as in `cv2` package. InsightFace models assume `RGB` color channel as in `PIL` package. So all our evaluation code uses `BGR` color channel with `cv2` package.



