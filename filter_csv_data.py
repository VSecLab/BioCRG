import os
import csv
import shutil

# Thresholds based on our previous analysis (25% of average)
ACTIVITY_THRESHOLDS = {
    'sphereActivity': 400.8 * 0.15,    
    'ladderActivity': 982.5 * 0.15,    
    'trashActivity': 1202.3 * 0.15,    
    'pilotActivity': 1066.8 * 0.15     
}

SOURCE_PATH = "/Users/grims/Documents/Research/Tesi/ML_tesi/data_logs/raw_original"
TARGET_PATH = "/Users/grims/Documents/Research/Tesi/ML_tesi/data_logs/raw_filtered3"

def create_directory_structure():
    """Create the raw_filtered directory with same structure as raw"""
    if os.path.exists(TARGET_PATH):
        shutil.rmtree(TARGET_PATH)
    
    # Create main directory
    os.makedirs(TARGET_PATH, exist_ok=True)
    
    # Create user directories
    for item in os.listdir(SOURCE_PATH):
        source_item = os.path.join(SOURCE_PATH, item)
        if os.path.isdir(source_item) and not item.startswith('.'):
            target_item = os.path.join(TARGET_PATH, item)
            os.makedirs(target_item, exist_ok=True)
            
            # Handle test_data subdirectories
            if item == "test_data":
                for subitem in os.listdir(source_item):
                    source_subitem = os.path.join(source_item, subitem)
                    if os.path.isdir(source_subitem) and not subitem.startswith('.'):
                        target_subitem = os.path.join(target_item, subitem)
                        os.makedirs(target_subitem, exist_ok=True)

def filter_csv_file(source_csv_path, target_csv_path):
    """Filter a single CSV file removing executions below threshold"""
    filtered_rows = []
    current_activity = None
    current_activity_rows = []
    current_activity_entries = 0
    
    with open(source_csv_path, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        filtered_rows.append(header)
        
        for row in reader:
            # Check for activity markers
            if len(row) >= 5 and row[4] == "ACTIVITY_START":
                # If we have a previous activity, decide whether to keep it
                if current_activity is not None:
                    threshold = ACTIVITY_THRESHOLDS.get(current_activity, 0)
                    if current_activity_entries >= threshold:
                        # Keep the previous activity
                        filtered_rows.extend(current_activity_rows)
                
                # Start new activity
                current_activity = row[2]
                current_activity_rows = [row]
                current_activity_entries = 0
                
            elif len(row) >= 5 and row[4] == "ACTIVITY_END":
                if current_activity is not None:
                    current_activity_rows.append(row)
                    # Check if we should keep this activity
                    threshold = ACTIVITY_THRESHOLDS.get(current_activity, 0)
                    if current_activity_entries >= threshold:
                        filtered_rows.extend(current_activity_rows)
                    else:
                        print(f"  Filtered out {current_activity} execution: {current_activity_entries} entries < {threshold:.1f} threshold")
                
                # Reset for next activity
                current_activity = None
                current_activity_rows = []
                current_activity_entries = 0
                
            else:
                # Regular row
                if current_activity is not None:
                    current_activity_rows.append(row)
                    current_activity_entries += 1
                else:
                    # Row outside of any activity (keep it)
                    filtered_rows.append(row)
        
        # Handle last activity if file ends without ACTIVITY_END
        if current_activity is not None:
            threshold = ACTIVITY_THRESHOLDS.get(current_activity, 0)
            if current_activity_entries >= threshold:
                filtered_rows.extend(current_activity_rows)
            else:
                print(f"  Filtered out incomplete {current_activity} execution: {current_activity_entries} entries < {threshold:.1f} threshold")
    
    # Write filtered data
    with open(target_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, lineterminator='\n')
        writer.writerows(filtered_rows)
    
    original_size = os.path.getsize(source_csv_path) / 1024
    filtered_size = os.path.getsize(target_csv_path) / 1024
    reduction = (1 - filtered_size / original_size) * 100
    
    print(f"  Original: {original_size:.1f} KB → Filtered: {filtered_size:.1f} KB (Reduced by {reduction:.1f}%)")

def process_all_csv_files():
    """Process all CSV files in the raw directory"""
    total_original_size = 0
    total_filtered_size = 0
    files_processed = 0
    
    for root, dirs, files in os.walk(SOURCE_PATH):
        for file in files:
            if file.endswith('.csv'):
                source_path = os.path.join(root, file)
                
                # Calculate relative path from SOURCE_PATH
                rel_path = os.path.relpath(source_path, SOURCE_PATH)
                target_path = os.path.join(TARGET_PATH, rel_path)
                
                print(f"Processing: {rel_path}")
                filter_csv_file(source_path, target_path)
                
                total_original_size += os.path.getsize(source_path)
                total_filtered_size += os.path.getsize(target_path)
                files_processed += 1
    
    total_original_mb = total_original_size / (1024 * 1024)
    total_filtered_mb = total_filtered_size / (1024 * 1024)
    total_reduction = (1 - total_filtered_size / total_original_size) * 100
    
    print(f"\n=== SUMMARY ===")
    print(f"Files processed: {files_processed}")
    print(f"Total original size: {total_original_mb:.2f} MB")
    print(f"Total filtered size: {total_filtered_mb:.2f} MB")
    print(f"Total size reduction: {total_reduction:.1f}%")
    print(f"\nActivity thresholds used:")
    for activity, threshold in ACTIVITY_THRESHOLDS.items():
        print(f"  {activity}: {threshold:.1f} entries ({threshold/50:.1f} seconds)")

def main():
    print("Creating directory structure...")
    create_directory_structure()
    
    print("Processing CSV files...")
    process_all_csv_files()
    
    print(f"\nFiltered data saved to: {TARGET_PATH}")

if __name__ == "__main__":
    main()