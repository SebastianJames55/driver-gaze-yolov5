python3 gaze_prediction.py \
        --data '/home/datasets/KITTI-360-low/data_2d_raw/2013_05_28_drive_0009_sync/image_00/data_rect/' \
        --features 'features/data_2d_raw/2013_05_28_drive_0009_sync/image_00/data_rect/' \
        --best 'grid1616_model_best.pth.tar' \
        -b 64 \
        --gridheight 16 \
        --gridwidth 16 \
        --yolo5bb 'runs/detect/data_2d_raw/2013_05_28_drive_0009_sync/image_00/data_rect/labels' \
        --visualizations 'outputs/2d_heatmaps/KITTI-360-low/data_2d_raw/2013_05_28_drive_0009_sync/image_00/data_rect/' \
        --threshhold 0.5 #\
        #--lstm \   
        #--convlstm \
        #--sequence 8
