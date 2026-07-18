(Following is currently only in dev branch of this repo) 

This repo is to run inference on KITTI-360 dataset. 

Original work - [Where and What: Driver Attention-based Object Detection](https://arxiv.org/pdf/2204.12150.pdf).

### Running Inference
The pretrained model can be tested with the following steps:

1. **Collect data:**  
Download Perspective Images for Train & Val (128G) images from [KITTI-360](https://www.cvlibs.net/datasets/kitti-360/user_login.php)
Sample placed in `datasets\`

2. **Extracting features:**  
Download weights (yolov5s.pt) from (yolov5 Release v5)[https://github.com/ultralytics/yolov5] and run ``extract_features.sh``
(extract_features.py is a modified version of detect.py)
This provides extracted features (`features/`), object bounding boxes (`runs/detect/`), and labels (`runs/detect/labels/`). Samples are provided in these directories.
Note: `models/` and `utils/` folders are taken from yolov5 release v5. A few functions and changes were taken from yolov5 release v6, v6.1

3. **Test:**  
For gaze map prediction and pixel-level/object-level evaluation run gaze_prediction.sh.  
Checkpoint for the trained grid 16x16 model (without LSTM) is available.      
Heatmap generated in `outputs/` (sample provided)

### References
- [Code for Saliency metrics](https://github.com/tarunsharma1/saliency_metrics/blob/master/salience_metrics.py)
- [Code for Convolutional LSTM](https://github.com/yaorong0921/Driver-Intention-Prediction/blob/master/models/convolution_lstm.py)
- [Code for extracting YOLOv5 features (Release 5.0, 6.0. 6.1)](https://github.com/ultralytics/yolov5)


### Citation
If you find this work useful or use the code, please cite as follows:

```
@article{rong2022and,
  title={Where and What: Driver Attention-based Object Detection},
  author={Rong, Yao and Kassautzki, Naemi-Rebecca and Fuhl, Wolfgang and Kasneci, Enkelejda},
  journal={arXiv preprint arXiv:2204.12150},
  year={2022}
}
```
