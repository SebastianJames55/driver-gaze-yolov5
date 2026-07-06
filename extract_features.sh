# Set the PYTHONPATH to include the yolov5 subdirectory dynamically
export PYTHONPATH="${PYTHONPATH}:$(pwd)/yolov5"

python3 extract_features.py \
  --source '/home/datasets/KITTI-360-low/data_2d_raw/2013_05_28_drive_0000_sync/image_00/data_rect' \
  --weights yolov5s.pt \
  --conf 0.25  \
  --save-txt \
  --save-conf \
  --features '/home/jamess/driver-gaze-yolov5/features/'
