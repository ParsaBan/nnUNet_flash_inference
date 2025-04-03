import os
import csv

def read_timing_results(file_path):
    timing_results = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines[1:]:  # Skip the header
            parts = line.strip().split(', ')
            timing_results.append(parts)
    return timing_results

def read_resource_usage(file_path):
    resource_usage = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines[1:]:  # Skip the header
            parts = line.strip().split(', ')
            resource_usage.append(parts)
    return resource_usage

def write_to_csv(data, headers, output_file):
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(data)

def main():
    base_dir = 'Inference_outputs/3d'
    timing_results_file = os.path.join(base_dir, 'timing_results.txt')
    resource_usage_file = os.path.join(base_dir, 'resource_usage.txt')
    
    timing_results = read_timing_results(timing_results_file)
    resource_usage = read_resource_usage(resource_usage_file)
    
    timing_results_headers = ['Task', 'Configuration', 'Time (seconds)', 'Number of Scans', 'Median Time per Scan (seconds)']
    resource_usage_headers = ['Task', 'Configuration', 'Min CPU (%)', 'Max CPU (%)', 'Avg CPU (%)', 'Min Memory (MB)', 'Max Memory (MB)', 'Avg Memory (MB)', 'Min Disk (MB)', 'Max Disk (MB)', 'Avg Disk (MB)']
    
    write_to_csv(timing_results, timing_results_headers, os.path.join(base_dir, 'timing_results.csv'))
    write_to_csv(resource_usage, resource_usage_headers, os.path.join(base_dir, 'resource_usage.csv'))

if __name__ == "__main__":
    main()