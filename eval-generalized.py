#!/usr/bin/env python3
"""
Evaluation script for the original authors code, which trains on the entire dataset instead of traning per-image
"""

import os
import argparse
import numpy as np
from skimage import io, img_as_float
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tabulate import tabulate

def save_results(metrics, save_path):
    """Save evaluation results to a text file with table formatting"""
    import csv
    from tabulate import tabulate
    
    results = []
    if metrics['psnr_values'] and metrics['ssim_values']:
        for img, psnr, ssim in zip(metrics['processed_files'], 
                                 metrics['psnr_values'], 
                                 metrics['ssim_values']):
            results.append([img, f"{psnr:.4f}", f"{ssim:.4f}"])
        
        # Add average row
        results.append(["---", "---", "---"])
        results.append([
            "Average", 
            f"{metrics['psnr_mean']:.4f} ± {metrics['psnr_std']:.4f}", 
            f"{metrics['ssim_mean']:.4f} ± {metrics['ssim_std']:.4f}"
        ])
    
    # Write to file
    with open(save_path, 'w', newline='') as f:
        # header
        f.write("Evaluation Results\n")
        f.write("==================\n\n")
        
        # metrics table
        f.write(tabulate(results, 
                        headers=["Image Name", "PSNR (dB)", "SSIM"], 
                        tablefmt="grid"))
        f.write("\n\n")
        
        f.write(f"Processed Images: {metrics['valid_pairs']}\n")
        f.write(f"Skipped/Errors: {len(metrics['problematic_files'])}\n")
        
        # problematic files
        if metrics['problematic_files']:
            f.write("\nProblematic Files:\n")
            for file in metrics['problematic_files']:
                f.write(f"- {file}\n")
                
        # CSV format if need for exports
        # writer = csv.writer(f)
        # writer.writerow(["Filename", "PSNR", "SSIM"])
        # for img, psnr, ssim in zip(metrics['processed_files'], 
        #                          metrics['psnr_values'], 
        #                          metrics['ssim_values']):
        #     writer.writerow([img, psnr, ssim])
        # writer.writerow([])
        # writer.writerow(["Average PSNR", metrics['psnr_mean']])
        # writer.writerow(["PSNR Std", metrics['psnr_std']])
        # writer.writerow(["Average SSIM", metrics['ssim_mean']])
        # writer.writerow(["SSIM Std", metrics['ssim_std']])

def evaluate_pairs(gt_folder, enhanced_folder, save_folder):
    results = []
    psnr_values = []
    ssim_values = []
    missing_files = []
    error_files = []
    metrics = {
        'psnr_values': [],
        'ssim_values': [],
        'processed_files': [],
        'problematic_files': [],
        'valid_pairs': 0
    }
    gt_files = sorted([f for f in os.listdir(gt_folder) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    for gt_file in gt_files:
        gt_path = os.path.join(gt_folder, gt_file)
        base, ext = os.path.splitext(gt_file)
        enhanced_file = f"{base}_enhance{ext}"
        enhanced_path = os.path.join(enhanced_folder, enhanced_file)

        if not os.path.exists(enhanced_path):
            missing_files.append(gt_file)
            metrics['problematic_files'].append(gt_file)
            print(f"{gt_file:<25} {'Missing':<12} {'Missing':<8}")
            continue

        try:
            # load and validate images
            gt_img = img_as_float(io.imread(gt_path))
            enh_img = img_as_float(io.imread(enhanced_path))
            
            if gt_img.shape != enh_img.shape:
                metrics['problematic_files'].append(gt_file)
                print(f"{gt_file:<25} {'Shape Mismatch':<12} {'Shape Mismatch':<8}")
                continue

            data_range = 1.0
            psnr = peak_signal_noise_ratio(gt_img, enh_img, data_range=data_range)
            ssim = structural_similarity(gt_img, enh_img, data_range=data_range, channel_axis=-1)           

            psnr_values.append(psnr)
            ssim_values.append(ssim)
            results.append([gt_file, f"{psnr:.4f}", f"{ssim:.4f}"])
            metrics['psnr_values'].append(psnr)
            metrics['ssim_values'].append(ssim)
            metrics['processed_files'].append(gt_file)
            metrics['valid_pairs']+=1

        except Exception as e:
            metrics['problematic_files'].append(gt_file)
            print(f"{gt_file:<25} {'Error':<12} {'Error':<8} ({str(e)})")

    metrics['psnr_mean'] = 0
    metrics['psnr_std'] = 0
    metrics['ssim_mean'] = 0
    metrics['ssim_std'] = 0
    # averages
    if metrics['psnr_values']:
        metrics['psnr_mean'] = np.mean(metrics['psnr_values'])
        metrics['psnr_std'] = np.std(metrics['psnr_values'])
        metrics['ssim_mean'] = np.mean(metrics['ssim_values'])
        metrics['ssim_std'] = np.std(metrics['ssim_values'])
    if save_folder:
        save_results(metrics, os.path.join(save_folder, "metrics.txt"))

    if psnr_values:
        avg_psnr = np.mean(psnr_values)
        std_psnr = np.std(psnr_values)
        avg_ssim = np.mean(ssim_values)
        std_ssim = np.std(ssim_values)
        results.append(["Average", f"{avg_psnr:.4f} ± {std_psnr:.4f}", f"{avg_ssim:.4f} ± {std_ssim:.4f}"])
    else:
        results.append(["No valid image pairs processed", "", ""])

    # error summary
    if missing_files or error_files:
        print("\nProcessing Issues:")
        if missing_files:
            print(f" - Missing enhanced images: {len(missing_files)}")
        if error_files:
            print(f" - Processing errors: {len(error_files)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ZERO-IG Metrics Evaluation')
    parser.add_argument('--gt_folder', required=True, help='Path to ground truth images')
    parser.add_argument('--enhanced_folder', required=True, help='Path to enhanced images')
    parser.add_argument('--save', type=str, default=None, help='Path to metric results')
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.gt_folder):
        raise FileNotFoundError(f"GT folder not found: {args.gt_folder}")
    if not os.path.isdir(args.enhanced_folder):
        raise FileNotFoundError(f"Enhanced folder not found: {args.enhanced_folder}")
    if not os.path.isdir(args.save) and args.save is not None:
        os.makedirs(args.save)

    evaluate_pairs(args.gt_folder, args.enhanced_folder, args.save)
