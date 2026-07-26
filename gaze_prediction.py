"""
The code for computing the saliency metrics is adapted from
https://github.com/tarunsharma1/saliency_metrics/blob/master/salience_metrics.py
"""

import os
import argparse
import time
import math

import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import functional as F
import torchvision

import numbers

import network
from kitti360 import Kitti360


parser = argparse.ArgumentParser(description='Feature Test')
parser.add_argument('--data', default='/home/datasets/KITTI-360-low/data_2d_raw/2013_05_28_drive_0009_sync/image_00/data_rect/', metavar='DIR', help='path to dataset')
parser.add_argument('--features', default='features/data_2d_raw/2013_05_28_drive_0009_sync/image_00/data_rect/', metavar='DIR', help='path to extracted features')
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
parser.add_argument('--yolo5bb', default='runs/detect/data_2d_raw/2013_05_28_drive_0009_sync/image_00/data_rect/labels', metavar='DIR', help='path to folder of yolo5 bounding box txt files')
parser.add_argument('--visualizations', default='outputs/2d_heatmaps/KITTI-360-low/data_2d_raw/2013_05_28_drive_0009_sync/image_00/data_rect/', metavar='DIR', help='path to folder for visalization of predicted gaze maps and target')
parser.add_argument('--threshhold', default=0.5, type=float, metavar='N', help='threshold for object-level evaluation')
parser.add_argument('--lstm', default=False, action='store_true', help='use lstm module')
parser.add_argument('--convlstm', default=False, action='store_true', help='use convlstm module')
parser.add_argument('--sequence', default=6, type=int, metavar='N', help='sequence length for lstm module')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    args = parser.parse_args()

    if torch.cuda.is_available():
        # Fallback to args.gpu if specified, otherwise default to device 0
        gpu_id = args.gpu if 'args' in locals() and hasattr(args, 'gpu') else 0
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")

    dim = args.gridwidth * args.gridheight
    th = 1/dim

    if args.gpu is not None:
        print("Use GPU: {} for testing".format(args.gpu))

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
            print("=> loading checkpoint '{}'".format(args.best))
            checkpoint = torch.load(args.best)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'], False)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.best, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.best))

    test(test_loader, model, args)


def test(test_loader, model, args):
    batch_time = AverageMeter()

    model.eval()

    all_count = 0

    hm_max_values = []

    i = 0

    heightfactor = 376 // args.gridheight
    widthfactor = 1408 // args.gridwidth

    smoothing = GaussianSmoothing(1, 5, 1).to(device)

    with torch.no_grad():
        end = time.time()
        for i, (input, img_names) in enumerate(test_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)

            output = torch.sigmoid(output)

            heatmap = grid2heatmap(output, [heightfactor, widthfactor], [args.gridheight, args.gridwidth])
            heatmap = F.interpolate(heatmap, size=[376, 1408], mode='bilinear', align_corners=False)
            heatmap = smoothing(heatmap)
            heatmap = F.pad(heatmap, (2, 2, 2, 2), mode='constant')
            heatmap = heatmap.view(heatmap.size(0), -1)
            heatmap = F.softmax(heatmap, dim=1)

            # normalize
            heatmap -= heatmap.min(1, keepdim=True)[0]
            heatmap /= heatmap.max(1, keepdim=True)[0]

            heatmap = heatmap.view(-1, 1, 376, 1408)

            for j in range(heatmap.size(0)):
                img_name = img_names[j]
                heatmap_img = heatmap[j] # predicted gaze map

                ##### compute object-level metrics

                filename  = os.path.join(args.yolo5bb, img_name + ".txt")

                if os.path.exists(filename):
                    with open(filename) as f:

                        for linestring in f:
                            all_count += 1

                            line = linestring.split()

                            width = float(line[3])
                            height = float(line[4])
                            x_center = float(line[1])
                            y_center = float(line[2])

                            x_min, x_max, y_min, y_max = bb_mapping(x_center, y_center, width, height)

                            # find maximum pixel value within object bounding box
                            heatmap_obj = heatmap_img[0, y_min : y_max + 1, x_min : x_max + 1]
                            heatmap_obj_max = torch.max(heatmap_obj)

                            # object is recognized if maximum pixel value is higher than th
                            hm_obj_recogn = heatmap_obj_max > args.threshhold

                            hm_max_values.append(heatmap_obj_max)

                        visualization(heatmap_img.cpu(), args.visualizations, img_name)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      .format(
                       i, len(test_loader), batch_time=batch_time))

        print('Test: [{0}/{1}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              .format(
               i, len(test_loader), batch_time=batch_time))


def bb_mapping(x_center_rel, y_center_rel, width_rel, height_rel, img_width = 1408, img_height = 376):
    """
    Compute absolute bounding boxes values for given image size and given relative parameters

    :param x_center_rel: relative x value of bb center
    :param y_center_rel: relative y value of bb center
    :param width_rel: relative width
    :param height_rel: relative height
    :return: absolute values of bb borders
    """
    width_abs = width_rel * img_width
    height_abs = height_rel * img_height
    x_center_abs = x_center_rel * img_width
    y_center_abs = y_center_rel * img_height
    x_min = int(math.floor(x_center_abs - 0.5 * width_abs))
    x_max = int(math.floor(x_center_abs + 0.5 * width_abs))
    y_min = int(math.floor(y_center_abs - 0.5 * height_abs))
    y_max = int(math.floor(y_center_abs + 0.5 * height_abs))
    bb = [x if x >= 0 else 0 for x in [x_min, x_max, y_min, y_max]]
    return bb


def grid2heatmap(grid, size, num_grid):
    """
    Rearrange and expand gridvector of size (gridheight * gridwidth) to size (376 x 1408) by duplicating values

    :param grid: output vector
    :param size: (H, W) of one expanded grid cell
    :param num_grids: (H, W) = grid dimension
    :return: 2D grid of size (376 x 1408)
    """
    new_heatmap = torch.zeros(grid.size(0), size[0] * num_grid[0], size[1] * num_grid[1])
    for i, item in enumerate(grid):
        idx = torch.nonzero(item)
        if idx.nelement() == 0:
            print('Empty')
            continue
        for x in idx:
            new_heatmap[i, x // num_grid[1] * size[0] : (x // num_grid[1] + 1) * size[0], x % num_grid[1] * size[1] : (x % num_grid[1] + 1) * size[1]] = item[x]
    output = new_heatmap.unsqueeze(1).to(device)

    return output


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
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

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
        return self.conv(input, weight=self.weight, groups=self.groups)


if __name__ == '__main__':
    main()
