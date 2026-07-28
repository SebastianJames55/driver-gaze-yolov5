"""
The code for computing the saliency metrics is adapted from
https://github.com/tarunsharma1/saliency_metrics/blob/master/salience_metrics.py
"""

import os
import argparse
import time
import math
import logging

import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import functional as F
import torchvision

import numbers

import network
from kitti360 import Kitti360


parser = argparse.ArgumentParser(description='Feature Test')
parser.add_argument('--data', default='/home/datasets/KITTI-360-low/data_2d_raw/2013_05_28_drive_0000_sync/image_00/data_rect/', metavar='DIR', help='path to dataset')
parser.add_argument('--features', default='features/data_2d_raw/2013_05_28_drive_0000_sync/image_00/data_rect/', metavar='DIR', help='path to extracted features')
parser.add_argument('--best', default='grid1616_model_best.pth.tar', type=str, metavar='PATH', help='path to best checkpoint (default: none)')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 128), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--gridheight', default=16, type=int, metavar='N',
                    help='number of rows in grid')
parser.add_argument('--gridwidth', default=16, type=int, metavar='N',
                    help='number of columns in grid ')
parser.add_argument('--yolo5bb', default='runs/detect/data_2d_raw/2013_05_28_drive_0000_sync/image_00/data_rect/labels', metavar='DIR', help='path to folder of yolo5 bounding box txt files')
parser.add_argument('--visualizations', default='outputs/2d_heatmaps/KITTI-360-low/data_2d_raw/2013_05_28_drive_0000_sync/image_00/data_rect/', metavar='DIR', help='path to folder for visalization of predicted gaze maps and target')
parser.add_argument('--threshhold', default=0.5, type=float, metavar='N', help='threshold for object-level evaluation')
parser.add_argument('--lstm', default=False, action='store_true', help='use lstm module')
parser.add_argument('--convlstm', default=False, action='store_true', help='use convlstm module')
parser.add_argument('--sequence', default=6, type=int, metavar='N', help='sequence length for lstm module')
parser.add_argument('--skip-existing', default=True, action='store_true',
                    help='skip visualization if pred file already exists')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize logger instance for this file
logger = logging.getLogger(__name__)


def main():
    logger.info("=== STARTING GAZE EVALUATION LOG ===")

    args = parser.parse_args()

    try:
        dim = args.gridwidth * args.gridheight
        th = 1/dim

        if args.gpu is not None:
            logger.info("Use GPU: {} for testing".format(args.gpu))

        model = network.Net(args.gridheight, args.gridwidth)

        if args.lstm:
            model = network.LstmNet(args.gridheight, args.gridwidth)

        if args.convlstm:
            model = network.ConvLSTMNet(args.gridheight, args.gridwidth, args.sequence)

        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
        
        testdir = args.data
        test_dataset = Kitti360("test", testdir, args.features,
                                th, (args.lstm or args.convlstm), args.sequence)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

        if args.best:
            if os.path.isfile(args.best):
                logger.info("=> loading checkpoint '{}'".format(args.best))
                checkpoint = torch.load(args.best)
                args.start_epoch = checkpoint['epoch']
                model.load_state_dict(checkpoint['state_dict'], False)
                logger.info("=> loaded checkpoint '{}' (epoch {})"
                    .format(args.best, checkpoint['epoch']))
            else:
                logger.warning("=> no checkpoint found at '{}'".format(args.best))

        test(test_loader, model, args)
    except Exception as e:
        logger.exception(f"Unhandled exception in main execution: {e}")
    finally:
        # Ensure all handlers flush log messages to the text file before exiting
        for handler in logging.getLogger().handlers:
            handler.flush()
    

def test(test_loader, model, args):
    batch_time = AverageMeter()

    # Configure logger to output both to console and a log file
    log_file_path = os.path.join(args.visualizations, "execution_log.txt")
    os.makedirs(args.visualizations, exist_ok=True)

    # Single, clean logging setup
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file_path, mode='a'),
            logging.StreamHandler()
        ],
        force=True  # Overrides any existing logger setups from PyTorch/Torchvision
    )

    logger = logging.getLogger(__name__)

    model.eval()

    i = 0

    smoothing = GaussianSmoothing(1, 5, 1).to(device)

    total_input_images = 0
    total_saved_preds = 0

    with torch.no_grad():
        end = time.time()
        for i, (input, img_names) in enumerate(test_loader):

            # Track batch input count
            batch_input_count = len(img_names)
            total_input_images += batch_input_count

            if args.skip_existing:
                # Filter out images that already have pred files
                new_input = []
                new_img_names = []

                for j, img_name in enumerate(img_names):
                    pred_path = os.path.join(args.visualizations, f"{img_name}_pred.png")
                    if os.path.exists(pred_path):
                        logger.info(f"[SKIP] {img_name} already visualized")
                    else:
                        new_input.append(input[j])
                        new_img_names.append(img_name)

                # If all images in this batch are already done → skip entire batch
                if len(new_input) == 0:
                    continue

                # Rebuild batch
                input = torch.stack(new_input)
                img_names = new_img_names
            
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            output = torch.sigmoid(output)

            # Reshape 1D grid directly to 2D low-res feature map (e.g., 16x16)
            heatmap = output.view(-1, 1, args.gridheight, args.gridwidth)

            # Smoothly upsample directly to target KITTI-360 resolution
            heatmap = F.interpolate(heatmap, size=[376, 1408], mode='bilinear', align_corners=False)

            # Smooth borders gracefully without hard constant padding
            heatmap = smoothing(heatmap)

            # Min-Max Normalization safely avoiding division by zero
            min_v = heatmap.amin(dim=(1, 2, 3), keepdim=True)
            max_v = heatmap.amax(dim=(1, 2, 3), keepdim=True)
            heatmap = (heatmap - min_v) / (max_v - min_v + 1e-8)

            batch_saved_count = 0

            for j in range(heatmap.size(0)):
                img_name = img_names[j]
                heatmap_img = heatmap[j] # predicted gaze map

                ##### compute object-level metrics

                filename  = os.path.join(args.yolo5bb, img_name + ".txt")

                if os.path.exists(filename):
                    with open(filename) as f:

                        if heatmap_img is None or heatmap_img.numel() == 0:
                            logger.warning(f"[WARN] Empty heatmap for {filename}")

                    try:
                        visualization(heatmap_img.cpu(), args.visualizations, img_name)
                        batch_saved_count += 1
                        total_saved_preds += 1
                    except Exception as e:
                        logger.error(f"[ERROR] Failed to save {filename}: {e}")
                else:
                    logger.warning(f"[WARN] No YOLO bounding box file for {filename}")    

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                logger.info(
                    f"Test: [{i}/{len(test_loader)}]\t"
                    f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    f"Batch Inputs: {batch_input_count} | Saved _pred: {batch_saved_count}"
                )

        logger.info("=" * 50)
        logger.info(f"PROCESSING SUMMARY:")
        logger.info(f"Total Input Images Processed: {total_input_images}")
        logger.info(f"Total _pred Files Saved:     {total_saved_preds}")
        logger.info(f"Failed / Skipped Files:       {total_input_images - total_saved_preds}")
        logger.info("=" * 50)


def visualization(heatmap, path, nr):
    heatmap = torchvision.transforms.functional.to_pil_image(heatmap)
    os.makedirs(path, exist_ok=True)
    heatmap.save(os.path.join(path, '%s_pred.png'%nr))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * torch.exp(-0.5 * ((mgrid - mean) / std) ** 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        # Calculate padding to keep output same size as input
        pad_h = (self.weight.shape[2] - 1) // 2
        pad_w = (self.weight.shape[3] - 1) // 2
        input_padded = F.pad(input, (pad_w, pad_w, pad_h, pad_h), mode='reflect')
        return self.conv(input_padded, weight=self.weight, groups=self.groups)


if __name__ == '__main__':
    main()
