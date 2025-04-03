import os
import subprocess
import time
import argparse
import psutil
import numpy as np

# Define the base directories
nnUNet_raw_data_base = "nnUNet_raw_data_base"
output_base_dir = "Inference_outputs"

# Define the task names and their corresponding folds (deemed optimal by DICE score)
task_names_and_folds = {
    1: ("BrainTumour", 0),
    2: ("Heart", 3),
    3: ("Liver", 0),
    4: ("Hippocampus", 2),
    5: ("Prostate", 0),
    6: ("Lung", 0),
    7: ("Pancreas", 2),
    8: ("HepaticVessel", 0),
    9: ("Spleen", 4),
    10: ("Colon", 4)
}

# Define the inference configurations
inference_configs = {
    "default": [],
    "fastest_no_mp": ["--disable_tta", "--mode", "fastest", "--disable_mixed_precision"],
    "flash": ["--disable_tta", "--mode", "fastest"],
    "no_tta": ["--disable_tta"],
    "no_tta_and_1_fold": ["--disable_tta"]
}

def entry_exists(file_path, task_name, config_name):
    """
    Check if an entry for the given task and configuration already exists in the file.

    Args:
        file_path (str): The path to the file.
        task_name (str): The name of the task.
        config_name (str): The name of the configuration.

    Returns:
        bool: True if the entry exists, False otherwise.
    """
    if not os.path.exists(file_path):
        return False

    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith(f"{task_name}, {config_name}"):
                return True
    return False

def monitor_resources():
    """
    Monitor and yield CPU, memory, and disk usage statistics.

    Yields:
        tuple: A tuple containing lists of CPU usage, memory usage, and disk usage statistics.
    """
    cpu_usage = []
    memory_usage = []
    disk_usage = []

    while True:
        cpu_usage.append(psutil.cpu_percent(interval=1))
        memory_usage.append(psutil.virtual_memory().used / (1024 * 1024))  # Convert to MB
        disk_usage.append(psutil.disk_io_counters().write_bytes / (1024 * 1024))  # Convert to MB

        yield cpu_usage, memory_usage, disk_usage

def calculate_median_scan_time(scan_timings_file):
    """
    Calculate the median scan time from the scan_timings.txt file.

    Args:
        scan_timings_file (str): The path to the scan_timings.txt file.

    Returns:
        float: The median scan time.
    """
    scan_times = []
    with open(scan_timings_file, 'r') as file:
        for line in file:
            time_str = line.split(': ')[-1].strip().split(' ')[0]
            scan_times.append(float(time_str))
    return np.median(scan_times)

def run_inference(task_id, task_name, fold, config_name, config_flags, input_dir, model_type):
    """
    Run the nnUNet inference for the specified task and configuration.

    Args:
        task_id (int): The ID of the task.
        task_name (str): The name of the task.
        fold (int): The fold number for the task.
        config_name (str): The name of the configuration.
        config_flags (list): The list of configuration flags.
        input_dir (str): The input directory containing the scans.
        model_type (str): The model type to use (2d or 3d).
    """
    # Map model_type to the correct nnUNet model type
    model_type_map = {
        "2d": "2d",
        "3d": "3d_fullres"
    }
    model_type_full = model_type_map[model_type]

    # Define the output directory based on the model type
    output_dir = os.path.join(output_base_dir, model_type, task_name, f"Pretrained_{task_name}_{config_name}/")
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Define the paths for the timing and resource usage files
    timing_file_path = os.path.join(output_base_dir, model_type, "timing_results.txt")
    resource_usage_file_path = os.path.join(output_base_dir, model_type, "resource_usage.txt")

    # Create the timing and resource usage files if they don't exist
    if not os.path.exists(timing_file_path):
        with open(timing_file_path, 'w') as timing_file:
            timing_file.write("Task, Configuration, Time (seconds), Number of Scans, Median Time per Scan (seconds)\n")

    if not os.path.exists(resource_usage_file_path):
        with open(resource_usage_file_path, 'w') as resource_file:
            resource_file.write("Task, Configuration, Min CPU (%), Max CPU (%), Avg CPU (%), Min Memory (MB), Max Memory (MB), Avg Memory (MB), Min Disk (MB), Max Disk (MB), Avg Disk (MB)\n")

    # Check if the entry already exists in the timing file
    if entry_exists(timing_file_path, task_name, config_name):
        print(f"Entry for Task {task_name} with configuration {config_name} already exists in the timing file. Skipping.")
        return

    # Count the number of scans in the input directory
    num_scans = len([f for f in os.listdir(input_dir) if f.endswith('.nii.gz')])
    
    # Construct the nnUNet_predict command
    command = [
        "nnUNet_predict",
        "-i", input_dir,
        "-o", output_dir,
        "-t", str(task_id),
        "-m", model_type_full
    ] + config_flags
    
    # Add the specific fold for the task
    if config_name in ["fastest_no_mp", "flash", "no_tta_and_1_fold"]:
        command += ["-f", str(fold)]
    
    # Print the start message
    print(f"Starting inference for Task {task_name} with configuration {config_name} using model {model_type_full}...")
    
    # Measure the time taken for inference
    start_time = time.time()
    
    # Start resource monitoring
    resource_monitor = monitor_resources()
    next(resource_monitor)
    
    # Run the inference command
    subprocess.run(command, check=True)
    
    # Stop resource monitoring
    cpu_usage, memory_usage, disk_usage = next(resource_monitor)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Calculate the median time per scan
    scan_timings_file = os.path.join(output_dir, "scan_timings.txt")
    median_time_per_scan = calculate_median_scan_time(scan_timings_file) if os.path.exists(scan_timings_file) else 0
    
    # Print the completion message
    print(f"Completed inference for Task {task_name} with configuration {config_name} in {elapsed_time:.2f} seconds.")
    
    # Save the timing result to the file
    with open(timing_file_path, 'a') as timing_file:
        timing_file.write(f"{task_name}, {config_name}, {elapsed_time:.2f}, {num_scans}, {median_time_per_scan:.2f}\n")
    
    # Calculate resource usage statistics
    min_cpu = min(cpu_usage)
    max_cpu = max(cpu_usage)
    avg_cpu = sum(cpu_usage) / len(cpu_usage)
    
    min_memory = min(memory_usage)
    max_memory = max(memory_usage)
    avg_memory = sum(memory_usage) / len(memory_usage)
    
    min_disk = min(disk_usage)
    max_disk = max(disk_usage)
    avg_disk = sum(disk_usage) / len(disk_usage)
    
    # Check if the entry already exists in the resource usage file
    if entry_exists(resource_usage_file_path, task_name, config_name):
        print(f"Entry for Task {task_name} with configuration {config_name} already exists in the resource usage file. Skipping.")
        return
    
    # Save the resource usage result to the file
    with open(resource_usage_file_path, 'a') as resource_file:
        resource_file.write(f"{task_name}, {config_name}, {min_cpu:.2f}, {max_cpu:.2f}, {avg_cpu:.2f}, {min_memory:.2f}, {max_memory:.2f}, {avg_memory:.2f}, {min_disk:.2f}, {max_disk:.2f}, {avg_disk:.2f}\n")

def main():
    """
    Main function to parse command line arguments and run the specified tasks and configurations.
    """
    parser = argparse.ArgumentParser(description="Run nnUNet inference on specified tasks and configurations.")
    parser.add_argument("-all", action="store_true", help="Run all configurations for all tasks.")
    parser.add_argument("-tasks", nargs='+', help="Specify the task IDs or names to run.")
    parser.add_argument("-configs", nargs='+', choices=inference_configs.keys(), help="Specify the configurations to run.")
    parser.add_argument("-list_tasks", action="store_true", help="List all task IDs and their corresponding names.")
    parser.add_argument("-input_dir", type=str, help="Specify a custom input directory. The data needs to follow the Medical Segmentation Decathlon format.")
    parser.add_argument("-model_type", choices=["2d", "3d"], required=True, help="Specify the model type to use (2d or 3d).")
    
    args = parser.parse_args()
    
    if args.list_tasks:
        print("Task IDs and their corresponding names:")
        for task_id, (task_name, _) in task_names_and_folds.items():
            print(f"{task_id}: {task_name}")
    elif args.all:
        # Run all configurations for all tasks
        for task_id, (task_name, fold) in task_names_and_folds.items():
            if task_name == "Liver":
                print(f"Skipping {task_name} as a whole")
                continue
            input_dir = args.input_dir if args.input_dir else os.path.join(nnUNet_raw_data_base, f"nnUNet_raw_data/Task00{task_id}_{task_name}/imagesTs/")
            for config_name, config_flags in inference_configs.items():
                if config_name == "default" and task_name == "Lung":
                    # Skip default configuration for Lung
                    with open(os.path.join(output_base_dir, args.model_type, "timing_results.txt"), 'a') as timing_file:
                        timing_file.write(f"{task_name}, {config_name}, N/A, N/A, N/A\n")
                    with open(os.path.join(output_base_dir, args.model_type, "resource_usage.txt"), 'a') as resource_file:
                        resource_file.write(f"{task_name}, {config_name}, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A\n")
                    print(f"Skipping default configuration for {task_name}")
                    continue
                print(f"Running {config_name} for task {task_name}")
                run_inference(task_id, task_name, fold, config_name, config_flags, input_dir, args.model_type)
    elif args.tasks:
        # Convert task names to task IDs if necessary
        task_ids = []
        for task in args.tasks:
            if task.isdigit():
                task_ids.append(int(task))
            else:
                for task_id, (task_name, _) in task_names_and_folds.items():
                    if task_name.lower() == task.lower():
                        task_ids.append(task_id)
                        break
        
        # Run the specified configurations for the specified tasks, or all configurations if no config is specified
        for task_id in task_ids:
            task_name, fold = task_names_and_folds[task_id]
            input_dir = args.input_dir if args.input_dir else os.path.join(nnUNet_raw_data_base, f"nnUNet_raw_data/Task00{task_id}_{task_name}/imagesTs/")
            if args.configs:
                for config_name in args.configs:
                    config_flags = inference_configs[config_name]
                    print(f"Running {config_name} for task {task_name}")
                    run_inference(task_id, task_name, fold, config_name, config_flags, input_dir, args.model_type)
            else:
                for config_name, config_flags in inference_configs.items():
                    if config_name == "default" and task_name == "Lung":
                        # Skip default configuration for Lung
                        with open(os.path.join(output_base_dir, args.model_type, "timing_results.txt"), 'a') as timing_file:
                            timing_file.write(f"{task_name}, {config_name}, N/A, N/A, N/A\n")
                        with open(os.path.join(output_base_dir, args.model_type, "resource_usage.txt"), 'a') as resource_file:
                            resource_file.write(f"{task_name}, {config_name}, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A\n")
                        print(f"Skipping default configuration for {task_name}")
                        continue
                    print(f"Running {config_name} for task {task_name}")
                    run_inference(task_id, task_name, fold, config_name, config_flags, input_dir, args.model_type)
    else:
        print("Please specify either -all or -tasks, or use -list_tasks to see available tasks.")

if __name__ == "__main__":
    main()