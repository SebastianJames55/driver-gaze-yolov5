#!/bin/bash

python3 gaze_prediction.py \
        --features '/home/jamess/driver-gaze-yolov5/features/data_2d_raw/2013_05_28_drive_0000_sync/image_00/data_rect/' \
        --model '/home/jamess/driver-gaze-yolov5/grid1616_model_best.pth.tar' \
        --images '/home/datasets/KITTI-360-low/data_2d_raw/2013_05_28_drive_0000_sync/image_00/data_rect/' \
        --visualizations '/home/jamess/driver-gaze-yolov5/outputs/2d_heatmaps/KITTI-360-low/data_2d_raw/2013_05_28_drive_0000_sync/image_00/data_rect/' \
        --gridheight 16 \
        --gridwidth 16 \
        -b 1 \
        --gpu 0 \
        # --lstm \
        # --sequence 6