#! /usr/bin/env bash

# run tuning script on multiple models
# chosing yolo_full, yolo_tile, and detr_full_frz
# also transfer tiled to comb_full dataset
rm -rf runs/detect/*tune*
# rm -rf runs/detect/*best

python3 ./src/tune_yolo.py yolo_full_train \
    --dataset comb_full \
    --project_dir /home/paperspace/seaberry_camille \
    --tune_epochs 5 \
    --train_epochs 50 \
    --iterations 10 \
    --train \
    --val 


# python3 ./src/tune_yolo.py detr_full_frz_train \
#     --dataset comb_full \
#     --model_type detr \
#     --project_dir /home/paperspace/seaberry_camille \
#     --tune_epochs 5 \
#     --train_epochs 50 \
#     --iterations 10 \
#     --train \
#     --val 

python3 ./src/tune_yolo.py yolo_tile_train \
    --dataset comb_tile \
    --project_dir /home/paperspace/seaberry_camille \
    --tune_epochs 5 \
    --train_epochs 50 \
    --iterations 10 \
    --train \
    --val \
    --label yolo_tile_notransfer

python3 ./src/tune_yolo.py yolo_tile_train \
    --dataset comb_full \
    --project_dir /home/paperspace/seaberry_camille \
    --tune_epochs 5 \
    --train_epochs 50 \
    --iterations 10 \
    --train \
    --val \
    --label yolo_tile_transfer
