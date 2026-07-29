"""
The following code is adapted from the file detect.py of https://github.com/ultralytics/yolov5 (Release 5.0)
"""

import argparse
import logging
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadStreams
from utils.general import (
    apply_classifier,
    check_img_size,
    check_imshow,
    check_requirements,
    non_max_suppression,
    save_one_box,
    scale_coords,
    set_logging,
    strip_optimizer,
    xyxy2xywh,
)
from utils.plots import colors, plot_one_box
from utils.torch_utils import load_classifier, select_device, time_synchronized

logging.basicConfig(
    filename='detect_errors.log',
    level=logging.ERROR,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)

activation = {}


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook


def detect(opt):
    source, weights, view_img, save_txt, imgsz = (
        opt.source,
        opt.weights,
        opt.view_img,
        opt.save_txt,
        opt.img_size,
    )
    save_img = not opt.nosave and not source.endswith('.txt')
    webcam = (
        source.isnumeric()
        or source.endswith('.txt')
        or source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    )
    features_dir = Path(opt.features)
    features_dir.mkdir(parents=True, exist_ok=True)

    save_dir = Path(opt.project) / opt.name
    save_dir.mkdir(parents=True, exist_ok=True)
    labels_dir = save_dir / 'labels'
    labels_dir.mkdir(parents=True, exist_ok=True)

    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'

    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())
    imgsz = check_img_size(imgsz, s=stride)
    names = model.module.names if hasattr(model, 'module') else model.names
    if half:
        model.half()

    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)
        modelc.load_state_dict(
            torch.load('weights/resnet101.pt', map_location=device)['model']
        ).to(device).eval()

    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    if device.type != 'cpu':
        model(
            torch.zeros(1, 3, imgsz, imgsz)
            .to(device)
            .type_as(next(model.parameters()))
        )

    # Register hook ONCE outside the main dataset loop to avoid memory leaks
    hook_handle = model.model[22].register_forward_hook(get_activation('after22'))

    # Counters
    total_input_files = 0
    saved_pt = 0
    saved_jpg = 0
    saved_txt_count = 0
    saved_no_det = 0

    skipped_pt = 0
    skipped_jpg = 0
    skipped_txt = 0
    skipped_all = 0

    vid_path, vid_writer = None, None
    t0 = time.time()

    for path, img, im0s, vid_cap in dataset:
        total_input_files += 1
        activation.clear()  # Clear stale feature maps from previous iteration

        imagename = Path(path[0] if webcam else path).stem
        pt_file = features_dir / f'{imagename}.pt'
        jpg_file = save_dir / f'{imagename}.jpg'
        txt_file = labels_dir / f'{imagename}.txt'

        if opt.skip_existing:
            # Check if outputs already exist (pt and jpg must exist; txt is optional if 0 detections)
            if pt_file.exists() and (opt.nosave or jpg_file.exists()):
                skipped_all += 1
                skipped_pt += 1
                if jpg_file.exists():
                    skipped_jpg += 1
                if txt_file.exists():
                    skipped_txt += 1
                print(f'Skipping {imagename} (already processed)')
                continue

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        t1 = time_synchronized()

        pred = model(img, augment=opt.augment)[0]

        if 'after22' in activation:
            tensor = activation['after22'].data.cpu()
            try:
                torch.save(tensor, pt_file)
                saved_pt += 1
            except Exception as e:
                logger.error(f'Failed saving PT for {imagename}: {e}')
        else:
            logger.error(f'Feature hook failed for {imagename}')

        if not pt_file.exists():
            logger.error(f'PT not saved for {imagename}')

        pred = non_max_suppression(
            pred,
            opt.conf_thres,
            opt.iou_thres,
            opt.classes,
            opt.agnostic_nms,
            max_det=opt.max_det,
        )
        t2 = time_synchronized()

        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        for i, det in enumerate(pred):
            if webcam:
                p, s, im0, frame = (
                    path[i],
                    f'{i}: ',
                    im0s[i].copy(),
                    dataset.count,
                )
            else:
                p, s, im0, frame = (
                    path,
                    '',
                    im0s.copy(),
                    getattr(dataset, 'frame', 0),
                )

            p = Path(p)
            save_path = str(save_dir / f'{imagename}.jpg')
            if dataset.mode == 'image':
                txt_path = labels_dir / imagename
            else:
                txt_path = labels_dir / f"{imagename}_{frame}"
            s += '%gx%g ' % img.shape[2:]
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            imc = im0.copy() if opt.save_crop else im0

            if len(det):
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape
                ).round()

                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                txt_file_written = False
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:
                        xywh = (
                            (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn)
                            .view(-1)
                            .tolist()
                        )
                        line = (
                            (cls, *xywh, conf)
                            if opt.save_conf
                            else (cls, *xywh)
                        )
                        try:
                            with open(f"{txt_path}.txt", 'a') as f:
                                f.write(
                                    ('%g ' * len(line)).rstrip() % line + '\n'
                                )
                            txt_file_written = True
                        except Exception as e:
                            logger.error(
                                f'Failed saving TXT for {p.name}: {e}'
                            )

                    if save_img or opt.save_crop or view_img:
                        c = int(cls)
                        label = (
                            None
                            if opt.hide_labels
                            else (
                                names[c]
                                if opt.hide_conf
                                else f'{names[c]} {conf:.2f}'
                            )
                        )
                        plot_one_box(
                            xyxy,
                            im0,
                            label=label,
                            color=colors(c, True),
                            line_thickness=opt.line_thickness,
                        )
                        if opt.save_crop:
                            save_one_box(
                                xyxy,
                                imc,
                                file=save_dir
                                / 'crops'
                                / names[c]
                                / f'{p.stem}.jpg',
                                BGR=True,
                            )

                if txt_file_written:
                    saved_txt_count += 1
            else:
                saved_no_det += 1
                logger.error(f'No detections for {p.name} at {p}')

            print(f'{s}Done. ({t2 - t1:.3f}s)')

            if save_img:
                try:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                        saved_jpg += 1
                    else:
                        if vid_path != save_path:
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()
                            if vid_cap:
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(
                                    vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                                )
                                h = int(
                                    vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                                )
                            else:
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                                save_path += '.mp4'
                            vid_writer = cv2.VideoWriter(
                                save_path,
                                cv2.VideoWriter_fourcc(*'mp4v'),
                                fps,
                                (w, h),
                            )
                        vid_writer.write(im0)
                        saved_jpg += 1
                except Exception as e:
                    logger.error(f'Failed saving JPG for {p.name}: {e}')

            if save_img and dataset.mode == 'image':
                if not Path(save_path).exists():
                    logger.error(f'JPG not saved for {p.name}')

            if save_txt and len(det):
                if not Path(f"{txt_path}.txt").exists():
                    logger.error(f'TXT not saved for {p.name}')

    # Remove hook clean-up
    hook_handle.remove()

    print('\nSUMMARY')
    print(f'Total input files: {total_input_files}')
    print(f'Saved PT: {saved_pt}')
    print(f'Saved JPG: {saved_jpg}')
    print(f'Saved TXT files: {saved_txt_count}')
    print(f'Files with NO detections: {saved_no_det}')
    print(f'Skipped (already processed): {skipped_all}')
    print(f'Skipped PT: {skipped_pt}')
    print(f'Skipped JPG: {skipped_jpg}')
    print(f'Skipped TXT: {skipped_txt}')

    if save_txt or save_img:
        s = (
            f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}"
            if save_txt
            else ''
        )
        print(f'Results saved to {save_dir}{s}')

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--weights',
        nargs='+',
        type=str,
        default='yolov5s.pt',
        help='model.pt path(s)',
    )
    parser.add_argument(
        '--source',
        type=str,
        default='datasets/KITTI-360-low/data_2d_raw/2013_05_28_drive_0000_sync/image_00/data_rect',
        help='source',
    )
    parser.add_argument(
        '--img-size', type=int, default=640, help='inference size (pixels)'
    )
    parser.add_argument(
        '--conf-thres',
        type=float,
        default=0.25,
        help='object confidence threshold',
    )
    parser.add_argument(
        '--iou-thres', type=float, default=0.45, help='IOU threshold for NMS'
    )
    parser.add_argument(
        '--max-det',
        type=int,
        default=1000,
        help='maximum number of detections per image',
    )
    parser.add_argument('--device', default='', help='cuda device')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument(
        '--save-txt', 
        action=argparse.BooleanOptionalAction,
        default=True,
        help='save results to *.txt (use --no-save-txt to disable)',
    )
    parser.add_argument(
        '--save-conf',
        action='store_true',
        help='save confidences in --save-txt labels',
    )
    parser.add_argument(
        '--save-crop',
        action='store_true',
        help='save cropped prediction boxes',
    )
    parser.add_argument(
        '--nosave', default=True, action='store_true', help='do not save images/videos'
    )
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument(
        '--agnostic-nms', action='store_true', help='class-agnostic NMS'
    )
    parser.add_argument(
        '--augment', action='store_true', help='augmented inference'
    )
    parser.add_argument(
        '--update', action='store_true', help='update all models'
    )
    parser.add_argument(
        '--project', default='runs/detect', help='save results to project/name'
    )
    parser.add_argument(
        '--name',
        default='data_2d_raw/2013_05_28_drive_0000_sync/image_00/data_rect',
        help='save results to project/name',
    )
    parser.add_argument(
        '--exist-ok', action='store_true', help='existing project/name ok'
    )
    parser.add_argument(
        '--line-thickness', default=3, type=int, help='bounding box thickness'
    )
    parser.add_argument(
        '--hide-labels',
        default=False,
        action='store_true',
        help='hide labels',
    )
    parser.add_argument(
        '--hide-conf',
        default=False,
        action='store_true',
        help='hide confidences',
    )
    parser.add_argument(
        '--features',
        default='features/data_2d_raw/2013_05_28_drive_0000_sync/image_00/data_rect',
        metavar='DIR',
        help='path to folder where to save features',
    )
    parser.add_argument(
        '--skip-existing',
        default=True,
        action='store_true',
        help='skip files already saved',
    )

    opt = parser.parse_args()
    print(opt)
    check_requirements(exclude=('tensorboard', 'pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:
            for opt.weights in [
                'yolov5s.pt',
                'yolov5m.pt',
                'yolov5l.pt',
                'yolov5x.pt',
            ]:
                detect(opt=opt)
                strip_optimizer(opt.weights)
        else:
            detect(opt=opt)