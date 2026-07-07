import os
import argparse
import time

import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import cv2
import numpy as np

# Repository network structure wrapper
import network

def parse_args():
    parser = argparse.ArgumentParser(description='KITTI360 Gaze Inference')

    # Paths for Inputs and Outputs
    parser.add_argument('--features', metavar='DIR', required=True, help='Path to extracted KITTI360 features directory')
    parser.add_argument('--model', default='', type=str, metavar='PATH', required=True, help='Path to pretrained model weight (.pth.tar)')
    parser.add_argument('--images', metavar='DIR', required=True, help='Path to raw KITTI360 image folder (used as background)')
    parser.add_argument('--visualizations', metavar='DIR', required=True, help='Path to folder where output heatmaps will be saved')

    # Grid Dimensions (Must match your pretrained model configuration, usually 16x16)
    parser.add_argument('--gridheight', default=16, type=int, metavar='N', help='number of rows in grid')
    parser.add_argument('--gridwidth', default=16, type=int, metavar='N', help='number of columns in grid')

    # Hardware / Batch Settings
    parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N', help='Mini-batch size')
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use (e.g. 0)')

    parser.add_argument('--lstm', default=False, action='store_true', help='use lstm module')
    parser.add_argument('--convlstm', default=False, action='store_true', help='use convlstm module')
    parser.add_argument('--sequence', default=6, type=int, metavar='N', help='sequence length for sequential models')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Device Setup
    if args.gpu is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
        
    os.makedirs(args.visualizations, exist_ok=True)

    # 2. Map Model Type String Expected by network.py
    model = network.Net(args.gridheight, args.gridwidth)

    if args.lstm:
        model = network.LstmNet(args.gridheight, args.gridwidth)

    if args.convlstm:
        model = network.ConvLSTMNet(args.gridheight, args.gridwidth, args.sequence)

    # 3. Load Pre-trained Repository Checkpoint Safely
    print(f"=> Loading weights from: {args.model}")
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    # 4. Read Pre-extracted Features (.npy matrices)
    feature_files = sorted([f for f in os.listdir(args.features) if f.endswith('.pt')])
    if len(feature_files) == 0:
        print(f"[-] No feature files found in {args.features}")
        return

    print(f"=> Processing {len(feature_files)} frames for inference...")

    with torch.no_grad():
        if args.lstm or args.convlstm:
            # --- Temporal Sequences Loop ---
            seq_len = args.sequence
            for i in range(len(feature_files) - seq_len + 1):
                window_files = feature_files[i : i + seq_len]
                
                print(f"[Processing] Frame {i+1}/{len(feature_files) - seq_len + 1}: {target_frame}...", end="\r")
                # Stack arrays across sequence axis -> Shape: (Sequence, Channels, H, W)
                sequence_tensors = []
                for f in window_files:
                   # 1. load the native serialized PyTorch file
                    try:
                        frame_tensor = torch.load(
                            os.path.join(args.features, f), 
                            map_location=device, 
                            weights_only=False
                        )
                    except Exception as e:
                        print(f"[-] Failed loading tensor frame {f}: {e}")
                        continue
                    
                    # Remove any unnecessary batch dimensions if your extractor saved them as 4D (1, C, H, W)
                    if frame_tensor.dim() == 4 and frame_tensor.size(0) == 1:
                        frame_tensor = frame_tensor.squeeze(0) # Reduce down to (C, H, W)
                        
                    sequence_tensors.append(frame_tensor)

                # Check if we successfully loaded the full required window length
                if len(sequence_tensors) != args.sequence:
                    continue

                # 2. Stack frames along a new timeline dimension (Dim 0)
                # Shape becomes: (Sequence, Channels, H, W)
                features_tensor = torch.stack(sequence_tensors, dim=0)
                
                # 3. Add the Batch axis (Dim 0) to achieve the mandatory 5D ConvLSTM shape
                # Shape becomes: (Batch=1, Sequence, Channels, H, W)
                features_tensor = features_tensor.unsqueeze(0).to(device)
                
                print(f"DEBUG - Raw features_tensor shape: {features_tensor.shape}")
                outputs = model(features_tensor)
                pred_grid = outputs.squeeze().cpu().numpy()
                
                # If model retains a sequence axis in its output, isolate the latest index frame
                if pred_grid.ndim > 2:
                    pred_grid = pred_grid[-1]

                # Match with the last frame of the temporal sequence window
                generate_visual_overlay(window_files[-1], pred_grid, args)
        else:
            # --- Static Single-Frame Loop ---
            for f_file in feature_files:
                print(f"[Processing] Frame {idx+1}/{len(feature_files)}: {f_file}...", end="\r")
                # Load the file natively with PyTorch, bypassing numpy completely
                try:
                    features_tensor = torch.load(
                        os.path.join(args.features, f_file), 
                        map_location=device, 
                        weights_only=False
                    )
                except Exception as e:
                    print(f"[-] Failed loading tensor block directly: {e}")
                    continue

                # If the extracted tensor doesn't have a batch/sequence dimension yet, 
                # ensure you unsqueeze it to match your model type input expectation:
                if features_tensor.dim() == 3: # e.g., (Channels, H, W) -> (1, Channels, H, W)
                    features_tensor = features_tensor.unsqueeze(0)
                
                features_tensor_batched = features_tensor.repeat(16, 1, 1, 1)
                outputs = model(features_tensor_batched)
                pred_flat = outputs[0].squeeze().cpu().numpy()
                pred_grid = pred_flat.reshape(args.gridheight, args.gridwidth)
                
                generate_visual_overlay(f_file, pred_grid, args)

    print(f"[+] Heatmap compilation completed successfully! Check outputs in: {args.visualizations}")

def generate_visual_overlay(feature_filename, pred_grid, args):
    """
    Extracts base ID matching from feature handles, handles background frames out of 
    args.images, and renders full 2D pixel attention heatmaps via OpenCV.
    """
    base_id = os.path.splitext(feature_filename)[0]
    
    # Try finding background asset versions (.png or .jpg formats) inside the image folder
    img_path = os.path.join(args.images, f"{base_id}.png")
    if not os.path.exists(img_path):
        img_path = os.path.join(args.images, f"{base_id}.jpg")
        if not os.path.exists(img_path):
            return  # Skip silently if matching background frame asset is not accessible

    orig_img = cv2.imread(img_path)
    img_h, img_w, _ = orig_img.shape

    # Upscale 16x16 grid prediction back into full resolution space using cubic mapping
    heatmap = cv2.resize(pred_grid, (img_w, img_h), interpolation=cv2.INTER_CUBIC)
    
    # Range validation normalize
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap_8bit = np.uint8(255 * heatmap)

    # Convert normalized matrix weights into a JET thermal color visualization palette
    color_heatmap = cv2.applyColorMap(heatmap_8bit, cv2.COLORMAP_JET)

    # Blended visualization output: 60% ambient scenery lighting, 40% eye focus map visibility
    blended_view = cv2.addWeighted(orig_img, 0.6, color_heatmap, 0.4, 0)
    
    # Save output visualization
    out_path = os.path.join(args.visualizations, f"{base_id}_heatmap.png")
    cv2.imwrite(out_path, blended_view)

if __name__ == '__main__':
    main()