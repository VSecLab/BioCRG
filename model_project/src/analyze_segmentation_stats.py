#!/usr/bin/env python3
"""
Script to analyze segmentation results and calculate statistics:
- Total number of segments
- Average number of segments per log
- Minimum number of segments found in a log
- Maximum number of segments found in a log
"""

import pandas as pd
import glob
import os
from pathlib import Path
import segmentation as seg
from data_processing.src import config

def analyze_segmentation_file(file_path):
    """Analyze a single segmentation results file"""
    print(f"\n=== Analyzing: {os.path.basename(file_path)} ===")
    
    df = pd.read_csv(file_path)
    
    # Get the number of segments for each log
    segments_per_log = df['NumberOfSegments'].tolist()
    
    # Calculate statistics
    total_segments = sum(segments_per_log)
    num_logs = len(segments_per_log)
    avg_segments = total_segments / num_logs if num_logs > 0 else 0
    min_segments = min(segments_per_log) if segments_per_log else 0
    max_segments = max(segments_per_log) if segments_per_log else 0
    
    print(f"Number of logs analyzed: {num_logs}")
    print(f"Total number of segments: {total_segments}")
    print(f"Average segments per log: {avg_segments:.2f}")
    print(f"Minimum segments in a log: {min_segments}")
    print(f"Maximum segments in a log: {max_segments}")
    
    # Additional details
    activity = df['Activity'].iloc[0] if not df.empty else "Unknown"
    threshold = df['Threshold'].iloc[0] if not df.empty else "Unknown"
    scaler = df['Scaler'].iloc[0] if not df.empty else "Unknown"
    
    print(f"Activity: {activity}")
    print(f"Threshold: {threshold}")
    print(f"Scaler: {scaler}")
    
    return {
        'file': os.path.basename(file_path),
        'activity': activity,
        'threshold': threshold,
        'scaler': scaler,
        'num_logs': num_logs,
        'total_segments': total_segments,
        'avg_segments': avg_segments,
        'min_segments': min_segments,
        'max_segments': max_segments,
        'segments_per_log': segments_per_log
    }

def analyze_all_segmentation_results():
    """Analyze all segmentation results files"""
    
    # Find all segmentation results files
    base_path = "/Users/grims/Documents/Research/Tesi/ML_tesi/data_logs/results/both_segmentation/segments_number"
    pattern = f"{base_path}/**/segmentation_results_*.csv"
    
    files = glob.glob(pattern, recursive=True)
    
    print(f"Found {len(files)} segmentation results files")
    
    all_results = []
    overall_segments = []
    
    for file_path in sorted(files):
        result = analyze_segmentation_file(file_path)
        all_results.append(result)
        overall_segments.extend(result['segments_per_log'])
    
    # Overall statistics across all files
    print(f"\n{'='*60}")
    print("OVERALL STATISTICS ACROSS ALL SEGMENTATION RESULTS")
    print(f"{'='*60}")
    
    total_files = len(all_results)
    total_logs_all_files = sum(r['num_logs'] for r in all_results)
    total_segments_all_files = sum(r['total_segments'] for r in all_results)
    
    if overall_segments:
        overall_avg = total_segments_all_files / total_logs_all_files if total_logs_all_files > 0 else 0
        overall_min = min(overall_segments)
        overall_max = max(overall_segments)
        
        print(f"Total files analyzed: {total_files}")
        print(f"Total logs across all files: {total_logs_all_files}")
        print(f"Total segments across all files: {total_segments_all_files}")
        print(f"Overall average segments per log: {overall_avg:.2f}")
        print(f"Overall minimum segments in a log: {overall_min}")
        print(f"Overall maximum segments in a log: {overall_max}")
    
    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY TABLE BY FILE")
    print(f"{'='*60}")
    print(f"{'File':<50} {'Logs':<6} {'Total':<6} {'Avg':<6} {'Min':<4} {'Max':<4}")
    print("-" * 80)
    
    for result in all_results:
        filename = result['file'][:47] + "..." if len(result['file']) > 50 else result['file']
        print(f"{filename:<50} {result['num_logs']:<6} {result['total_segments']:<6} {result['avg_segments']:<6.1f} {result['min_segments']:<4} {result['max_segments']:<4}")


def main(): 

    target_activity = "ladderActivity" 
    threshold = 0.6
    scaler = "standard"
    features = config.FEATURES
    segmentation_result = config.BOTH_SEGMENTATION_DIR 

    for th in [0.6, 0.65, 0.7, 0.75]:
        for activity in ["sphereActivity", "ladderActivity", "trashActivity", "pilotActivity"]:
            print(f"\n=== Segmenting activity: {target_activity} with threshold: {th} and scaler: {scaler} ===")
            _, _ = seg.segment_all_users(
                    activity=activity,
                    features=features,
                    threshold=th,
                    scaler=scaler,
                    filepath=segmentation_result
                )
            
    return 


if __name__ == "__main__":
    main()
    analyze_all_segmentation_results()