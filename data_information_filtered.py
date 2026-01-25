import os
import csv

DATA_PATH = "/Users/grims/Documents/Research/Tesi/ML_tesi/data_logs/raw"

def get_csv_files(root_path):
    csv_files = []
    for dirpath, _, filenames in os.walk(root_path):
        for file in filenames:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(dirpath, file))
    return csv_files

def analyze_csv(csv_path):
    file_size_kb = os.path.getsize(csv_path) / 1024
    entry_count = 0
    activity_counts = {}
    activity_instances = {}
    activity_entries = {}  # Track entries per execution: {user: {activity: [entries_run1, entries_run2, ...]}}
    current_activity = None
    current_activity_entries = 0
    current_user = os.path.basename(os.path.dirname(csv_path))
    activity_col_idx = None

    with open(csv_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        if "Activity" in header:
            activity_col_idx = header.index("Activity")
        else:
            activity_col_idx = None

        for row in reader:
            entry_count += 1
            
            # Marker handling
            if len(row) >= 5 and row[4] == "ACTIVITY_START":
                current_activity = row[2]
                current_activity_entries = 0
                activity_instances.setdefault(current_user, {}).setdefault(current_activity, 0)
                activity_instances[current_user][current_activity] += 1
                
                # Initialize entries tracking structure
                activity_entries.setdefault(current_user, {}).setdefault(current_activity, [])
                
            elif len(row) >= 5 and row[4] == "ACTIVITY_END":
                if current_activity is not None:
                    # Save the number of entries for this execution
                    activity_entries[current_user][current_activity].append(current_activity_entries)
                current_activity = None
                current_activity_entries = 0
            
            # Count entries during activity execution
            if current_activity is not None:
                current_activity_entries += 1

            # Count activities from Activity column (for backward compatibility)
            if activity_col_idx is not None and row[activity_col_idx]:
                act = row[activity_col_idx]
                activity_counts[act] = activity_counts.get(act, 0) + 1

    return {
        "file": csv_path,
        "size_kb": file_size_kb,
        "entries": entry_count,
        "activity_instances": activity_instances,
        "activity_entries": activity_entries,
        "activity_counts": activity_counts
    }

def merge_activity_instances(all_instances):
    merged = {}
    for user_dict in all_instances:
        for user, acts in user_dict.items():
            if user not in merged:
                merged[user] = {}
            for act, count in acts.items():
                merged[user][act] = merged[user].get(act, 0) + count
    return merged

def merge_activity_entries(all_entries):
    merged = {}
    for user_dict in all_entries:
        for user, acts in user_dict.items():
            if user not in merged:
                merged[user] = {}
            for act, entries_list in acts.items():
                if act not in merged[user]:
                    merged[user][act] = []
                merged[user][act].extend(entries_list)
    return merged

def calculate_total_executions(merged_instances):
    """Calculate total executions per activity across all users"""
    total_executions = {}
    for user, acts in merged_instances.items():
        for act, count in acts.items():
            total_executions[act] = total_executions.get(act, 0) + count
    return total_executions

def calculate_activity_averages(merged_entries):
    """Calculate average entries per execution for each activity across all users"""
    activity_averages = {}
    
    for user, acts in merged_entries.items():
        for act, entries_list in acts.items():
            if act not in activity_averages:
                activity_averages[act] = []
            activity_averages[act].extend(entries_list)
    
    # Calculate averages and time estimates
    averages = {}
    sampling_rate = 50  # entries per second
    
    for act, all_entries in activity_averages.items():
        if all_entries:
            avg_entries = sum(all_entries) / len(all_entries)
            avg_time_seconds = avg_entries / sampling_rate
            min_time_seconds = min(all_entries) / sampling_rate
            max_time_seconds = max(all_entries) / sampling_rate
            
            averages[act] = {
                'average': avg_entries,
                'total_executions': len(all_entries),
                'min_entries': min(all_entries),
                'max_entries': max(all_entries),
                'avg_time_seconds': avg_time_seconds,
                'avg_time_minutes': avg_time_seconds / 60,
                'min_time_seconds': min_time_seconds,
                'max_time_seconds': max_time_seconds
            }
    
    return averages

def merge_activity_counts(all_counts):
    merged = {}
    for act_dict in all_counts:
        for act, count in act_dict.items():
            merged[act] = merged.get(act, 0) + count
    return merged

def main():
    csv_files = get_csv_files(DATA_PATH)
    all_activity_instances = []
    all_activity_entries = []
    all_activity_counts = []
    
    print("=== ANALYSIS OF RAW DATA (NO FILTERING) ===")
    print(f"Data source: {DATA_PATH}")
    print("\nFile summary:")
    
    for csv_file in csv_files:
        info = analyze_csv(csv_file)
        print(f"{os.path.basename(csv_file)}: {info['size_kb']:.2f} KB, {info['entries']} entries")
        all_activity_instances.append(info["activity_instances"])
        all_activity_entries.append(info["activity_entries"])
        all_activity_counts.append(info["activity_counts"])

    merged_instances = merge_activity_instances(all_activity_instances)
    merged_entries = merge_activity_entries(all_activity_entries)
    total_executions = calculate_total_executions(merged_instances)
    activity_averages = calculate_activity_averages(merged_entries)
    merged_counts = merge_activity_counts(all_activity_counts)

    print("\nActivity executions per user:")
    for user, acts in merged_instances.items():
        print(f"User: {user}")
        for act, count in acts.items():
            print(f"  {act}: {count} times")

    print("\nTotal executions per activity (all users):")
    for act, count in total_executions.items():
        print(f"{act}: {count} executions")

    print("\nAverage entries per execution by activity (raw data, no filtering):")
    for act, stats in activity_averages.items():
        print(f"{act}:")
        print(f"  Average entries per execution: {stats['average']:.1f}")
        print(f"  Average execution time: {stats['avg_time_seconds']:.1f} seconds ({stats['avg_time_minutes']:.2f} minutes)")
        print(f"  Total executions: {stats['total_executions']}")
        print(f"  Min entries in single execution: {stats['min_entries']} ({stats['min_time_seconds']:.1f} seconds)")
        print(f"  Max entries in single execution: {stats['max_entries']} ({stats['max_time_seconds']:.1f} seconds)")
        print()

    print("\nEntries per execution for each user and activity:")
    for user, acts in merged_entries.items():
        print(f"\nUser: {user}")
        for act, entries_list in acts.items():
            print(f"  Activity: {act}")
            for i, entries in enumerate(entries_list, 1):
                print(f"    Execution {i}: {entries} entries")
            if entries_list:
                avg_entries = sum(entries_list) / len(entries_list)
                print(f"    Average entries per execution: {avg_entries:.1f}")

    print("\nLegacy activity counts (from Activity column):")
    for act, count in merged_counts.items():
        print(f"{act}: {count} times")

if __name__ == "__main__":
    main()