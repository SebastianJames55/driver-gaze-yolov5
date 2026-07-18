python3 gaze_prediction.py \
        --data 'C:/Users/SEBASTIAN/OneDrive/Documents/ALU/SS2026/DL lab/project/RL3/datasets/KITTI-360-low/data_2d_raw/2013_05_28_drive_0000_sync/image_00/data_rect/' \
        --features 'C:/Users/SEBASTIAN/OneDrive/Documents/ALU/SS2026/DL lab/project/RL3/driver-gaze-yolov5/features/data_2d_raw/2013_05_28_drive_0000_sync/image_00/data_rect/' \
        --best 'C:/Users/SEBASTIAN/OneDrive/Documents/ALU/SS2026/DL lab/project/RL3/driver-gaze-yolov5/grid1616_model_best.pth.tar' \
        -b 64 \
        --gridheight 16 \
        --gridwidth 16 \
        --yolo5bb 'C:/Users/SEBASTIAN/OneDrive/Documents/ALU/SS2026/DL lab/project/RL3/driver-gaze-yolov5/runs/detect/data_2d_raw/2013_05_28_drive_0000_sync/image_00/data_rect/labels' \
        --visualizations 'C:/Users/SEBASTIAN/OneDrive/Documents/ALU/SS2026/DL lab/project/RL3/driver-gaze-yolov5/outputs/2d_heatmaps/KITTI-360-low/data_2d_raw/2013_05_28_drive_0000_sync/image_00/data_rect/' \
        --threshhold 0.5 #\
        #--lstm \   
        #--convlstm \
        #--sequence 8
