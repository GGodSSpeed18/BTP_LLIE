"""
Evaluation Script for ZERO-IG Results

Expected directory structures:

Ground-truth (data_root):
  data_root/
    1/
      high/
        i1.png      <-- ground truth
      low/
        i1.png
    2/
      high/
        i2.png
      low/
        i2.png
    ...

Enhanced Results (results_root):
  results_root/
    1/
      i1_denoise.png
      i1_enhance.png   <-- enhanced output for evaluation
    2/
      i2_denoise.png
      i2_enhance.png
    ...

This script computes PSNR and SSIM between each ground-truth image in the "high" folder (from data_root)
and its corresponding enhanced image (with the "_enhance" suffix) in the corresponding folder in results_root.
The aggregated metrics are then saved as a readable table and CSV in a specified save folder.
"""

import os
import argparse
import numpy as np
from skimage import io, img_as_float
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tabulate import tabulate
import csv

def save_results(metrics, save_path):
    """
    Save evaluation results to a text file.
    The file contains a formatted table of metrics and a CSV dump for further parsing.
    """
    results = []
    if metrics['psnr_values'] and metrics['ssim_values']:
        for img, psnr, ssim in zip(metrics['processed_files'], metrics['psnr_values'], metrics['ssim_values']):
            results.append([img, f"{psnr:.4f}", f"{ssim:.4f}"])
        results.append(["---", "---", "---"])
        results.append(["Average", 
                        f"{metrics['psnr_mean']:.4f} ± {metrics['psnr_std']:.4f}", 
                        f"{metrics['ssim_mean']:.4f} ± {metrics['ssim_std']:.4f}"])
    
    with open(save_path, 'w', newline='') as f:
        f.write("Evaluation Results\n")
        f.write("==================\n\n")
        f.write(tabulate(results, headers=["Image Name", "PSNR (dB)", "SSIM"], tablefmt="grid"))
        f.write("\n\n")
        f.write(f"Processed Images: {metrics['valid_pairs']}\n")
        f.write(f"Skipped/Errors: {len(metrics['problematic_files'])}\n")
        if metrics['problematic_files']:
            f.write("\nIssues Encountered:\n")
            for issue in metrics['problematic_files']:
                f.write(f"- {issue}\n")
        f.write("\n\nCSV Format:\n")
        writer = csv.writer(f)
        writer.writerow(["Filename", "PSNR", "SSIM"])
        for img, psnr, ssim in zip(metrics['processed_files'], metrics['psnr_values'], metrics['ssim_values']):
            writer.writerow([img, psnr, ssim])
        writer.writerow([])
        writer.writerow(["Average PSNR", metrics['psnr_mean']])
        writer.writerow(["PSNR Std", metrics['psnr_std']])
        writer.writerow(["Average SSIM", metrics['ssim_mean']])
        writer.writerow(["SSIM Std", metrics['ssim_std']])

def evaluate_images(data_root, results_root, save_folder):

    metrics = {
        'psnr_values': [],
        'ssim_values': [],
        'processed_files': [],
        'problematic_files': [],
        'valid_pairs': 0,
    }
    
    # sorted list of numbered directories in data_root
    numbered_dirs = sorted([d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))],
                            key=lambda x: int(x))
    
    for num_dir in numbered_dirs:
        # build paths for ground truth and corresponding enhanced results
        gt_dir = os.path.join(data_root, num_dir, "high")
        if not os.path.isdir(gt_dir):
            metrics['problematic_files'].append(f"{num_dir}: Missing 'high' folder")
            continue
        gt_files = [f for f in os.listdir(gt_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not gt_files:
            metrics['problematic_files'].append(f"{num_dir}: No ground truth image found in 'high'")
            continue
        gt_file = gt_files[0]
        gt_path = os.path.join(gt_dir, gt_file)
        
        # expected enhanced filename:  "img_enhance"
        base, ext = os.path.splitext(gt_file)
        enh_file = f"{base}_enhance{ext}"
        enh_path = os.path.join(results_root, num_dir, enh_file)
        if not os.path.exists(enh_path):
            metrics['problematic_files'].append(f"{num_dir}: Enhanced file not found ({enh_file})")
            continue
        
        # load images as floats (normalized to [0, 1])
        try:
            gt_img = img_as_float(io.imread(gt_path))
            enh_img = img_as_float(io.imread(enh_path))
        except Exception as e:
            metrics['problematic_files'].append(f"{num_dir}: Error reading images ({str(e)})")
            continue
        
        # check that shapes match
        if gt_img.shape != enh_img.shape:
            metrics['problematic_files'].append(f"{num_dir}: Shape mismatch between ground truth and enhanced images")
            continue
        
        data_range = 1.0
        try:
            psnr_val = peak_signal_noise_ratio(gt_img, enh_img, data_range=data_range)
            # For RGB images (channel_axis=-1 for the last dimension)
            ssim_val = structural_similarity(gt_img, enh_img, data_range=data_range, channel_axis=-1)
        except Exception as e:
            metrics['problematic_files'].append(f"{num_dir}: Error computing metrics ({str(e)})")
            continue
        
        metrics['psnr_values'].append(psnr_val)
        metrics['ssim_values'].append(ssim_val)
        metrics['processed_files'].append(gt_file)
        metrics['valid_pairs'] += 1
        print(f"{num_dir}: {gt_file} -> PSNR: {psnr_val:.4f}, SSIM: {ssim_val:.4f}")
    
    # compute averages
    if metrics['psnr_values']:
        metrics['psnr_mean'] = np.mean(metrics['psnr_values'])
        metrics['psnr_std'] = np.std(metrics['psnr_values'])
        metrics['ssim_mean'] = np.mean(metrics['ssim_values'])
        metrics['ssim_std'] = np.std(metrics['ssim_values'])
    else:
        metrics['psnr_mean'] = metrics['psnr_std'] = metrics['ssim_mean'] = metrics['ssim_std'] = 0
    
    if save_folder:
        os.makedirs(save_folder, exist_ok=True)
        save_results(metrics, os.path.join(save_folder, "metrics.txt"))
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="ZERO-IG Evaluation Script")
    parser.add_argument('--data_root', type=str, required=True,
                        help="Root directory containing numbered folders with 'high' (GT) and 'low' images.")
    parser.add_argument('--results_root', type=str, required=True,
                        help="Root directory containing enhanced images (each numbered folder with *_enhance images).")
    parser.add_argument('--save', type=str, default=None,
                        help="Directory where evaluation metrics will be saved.")
    args = parser.parse_args()
    
    if not os.path.isdir(args.data_root):
        raise FileNotFoundError(f"Data root not found: {args.data_root}")
    if not os.path.isdir(args.results_root):
        raise FileNotFoundError(f"Results root not found: {args.results_root}")
    
    metrics = evaluate_images(args.data_root, args.results_root, args.save)
    
    print("\nFinal Evaluation Metrics:")
    print(f"Processed Images: {metrics['valid_pairs']}")
    if metrics['valid_pairs'] > 0:
        print(f"Average PSNR: {metrics['psnr_mean']:.4f} ± {metrics['psnr_std']:.4f}")
        print(f"Average SSIM: {metrics['ssim_mean']:.4f} ± {metrics['ssim_std']:.4f}")
    else:
        print("No valid image pairs processed.")
    
    if metrics['problematic_files']:
        print("Issues encountered:")
        for issue in metrics['problematic_files']:
            print(" -", issue)

if __name__ == "__main__":
    main()
