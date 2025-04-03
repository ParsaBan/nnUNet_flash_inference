import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import argparse
import seaborn as sns
import pandas as pd

def load_nifti(file_path):
    img = nib.load(file_path)
    data = img.get_fdata()
    header = img.header
    voxel_dims = header.get_zooms()  # Get voxel dimensions in mm
    voxel_volume = np.prod(voxel_dims)  # Compute voxel volume in cubic mm
    return data, voxel_volume

def compute_volume(segmentation, voxel_volume):
    return np.sum(segmentation > 0) * voxel_volume

def compare_volumes(task_folder_path):
    volumes = {}
    for config_folder in os.listdir(task_folder_path):
        config_folder_path = os.path.join(task_folder_path, config_folder)
        if os.path.isdir(config_folder_path):
            volumes[config_folder] = []
            for file_name in os.listdir(config_folder_path):
                if file_name.endswith('.nii.gz'):
                    file_path = os.path.join(config_folder_path, file_name)
                    segmentation, voxel_volume = load_nifti(file_path)
                    volume = compute_volume(segmentation, voxel_volume)
                    volumes[config_folder].append((file_name, volume))
    return volumes

def compute_statistics(volumes):
    stats = {}
    for folder_name, volume_list in volumes.items():
        if not volume_list:
            print(f"No volumes found for {folder_name}. Skipping statistics computation.")
            continue
        volumes_only = [volume for _, volume in volume_list]
        median_volume = np.median(volumes_only)
        std_volume = np.std(volumes_only)
        stats[folder_name] = {
            'median_volume': median_volume,
            'std_volume': std_volume,
            'min_volume': np.min(volumes_only),
            'max_volume': np.max(volumes_only)
        }
    return stats

def plot_statistics(stats, output_dir):
    if not stats:
        print("No statistics to plot.")
        return
    
    labels = [format_config_name(label)
              .replace('Fastest No Mp', 'Flash No MP')
              .replace('No Tta', 'No TTA')
              .replace('No Tta and 1 Fold', 'No TTA and 1 Fold') for label in stats.keys()]
    median_volumes = [stat['median_volume'] for stat in stats.values()]
    std_volumes = [stat['std_volume'] for stat in stats.values()]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(14, 8))
    rects1 = ax.bar(x - width/2, median_volumes, width, label='Median Volume', color='skyblue', edgecolor='black')
    rects2 = ax.bar(x + width/2, std_volumes, width, label='Standard Deviation', color='lightgreen', edgecolor='black')
    
    ax.set_xlabel('Inference Rounds', fontsize=14)
    ax.set_ylabel('Volume (mm^3)', fontsize=14)
    ax.set_title('Median and Standard Deviation of Volumes for Different Inference Rounds', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=12)
    ax.legend(fontsize=12)
    
    # Add gridlines for better readability
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add value labels on top of the bars
    def add_labels(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)
    
    add_labels(rects1)
    add_labels(rects2)
    
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, 'volume_statistics.png'))
    plt.close()

def compute_dice_score(segmentation, ground_truth):
    if segmentation.shape != ground_truth.shape:
        print(f"Shape mismatch: segmentation shape {segmentation.shape}, ground truth shape {ground_truth.shape}")
        return 0.0
    intersection = np.sum((segmentation > 0) & (ground_truth > 0))
    volume_sum = np.sum(segmentation > 0) + np.sum(ground_truth > 0)
    if volume_sum == 0:
        return 1.0  # Both are empty
    return 2.0 * intersection / volume_sum

def load_scan_timings(task_folder_path):
    scan_timings = {}
    for config_folder in os.listdir(task_folder_path):
        config_folder_path = os.path.join(task_folder_path, config_folder)
        scan_timings_file = os.path.join(config_folder_path, 'scan_timings.txt')
        if os.path.isfile(scan_timings_file):
            with open(scan_timings_file, 'r') as f:
                timings = []
                for line in f:
                    time_str = line.split(': ')[-1].strip().split(' ')[0]
                    timings.append(float(time_str))
                scan_timings[config_folder] = timings
    return scan_timings

def format_config_name(config_name):
    return config_name.replace('_', ' ').replace('Pretrained ', '').replace(' no tta and ', ' No TTA and ').title()

def plot_scan_timings_boxplot(scan_timings, output_dir, config_name):
    data = []
    for timing in scan_timings:
        data.append(timing)
    df = pd.DataFrame(data, columns=['Inference Time (seconds)'])
    plt.figure(figsize=(14, 8))
    sns.boxplot(y='Inference Time (seconds)', data=df)
    if not df['Inference Time (seconds)'].isnull().all():
        plt.ylim(0, df['Inference Time (seconds)'].quantile(0.95))
    title = (format_config_name(config_name)
             .replace('Fastest No Mp', 'Flash No MP')
             .replace('No Tta', 'No TTA')
             .replace('No Tta and 1 Fold', 'No TTA and 1 Fold'))
    plt.ylabel('Inference Time (seconds)', fontsize=14)
    plt.title(f'Scan Timings Distribution for {title}', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{config_name}_scan_timings_boxplot.png'))
    plt.close()

def plot_inference_times_stacked_lineplot(scan_timings, output_dir, title):
    plt.figure(figsize=(14, 8))
    color_map = {
        'default': 'blue',
        'no_tta': 'green',
        'no_tta_and_1_fold': 'orange',
        'flash': 'red',
        'fastest_no_mp': 'purple'
    }
    
    total_points = 0
    for config, timings in scan_timings.items():
        if len(timings) == 0:
            continue
        
        # Extract base config name by removing "Pretrained_{Task}_" prefix
        base_config = "_".join(config.split("_")[2:])
        total_points += len(timings)
        
        timings = [t for t in timings if t >= 0]
        label = (format_config_name(config)
                 .replace('Pretrained ', '')
                 .replace('Fastest No Mp', 'Flash No MP')
                 .replace('No Tta and 1 Fold', 'No TTA and 1 Fold')
                 .replace('No Tta', 'No TTA'))
        color = color_map.get(base_config)
        
        if color is None:
            raise ValueError(f"Missing color mapping for config: {base_config} (original: {config})")
            
        plt.plot(range(len(timings)), sorted(timings), label=label, color=color)

    title_with_count = f"{title}  (n={round(total_points/5)})"
    plt.xlabel('Inference Instance', fontsize=14)
    plt.ylabel('Inference Time (seconds)', fontsize=14)
    plt.title(title_with_count, fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'inference_times_stacked_lineplot.png'))
    plt.close()

def plot_median_time_vs_volume(scan_timings, volumes, output_dir, title):
    plt.figure(figsize=(14, 8))
    for config, timings in scan_timings.items():
        if len(timings) == 0 or config not in volumes:
            continue
        median_time = np.median(timings)
        lower_bound = np.percentile(timings, 25)
        upper_bound = np.percentile(timings, 75)
        mean_volume = np.mean([v for _, v in volumes[config]])
        plt.errorbar(mean_volume, median_time, yerr=[[median_time - lower_bound], [upper_bound - median_time]], fmt='o', label=format_config_name(config))
    plt.xlabel('Mean Volume (mm^3)', fontsize=14)
    plt.ylabel('Median Inference Time (seconds)', fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'median_time_vs_volume.png'))
    plt.close()

def plot_combined_median_time_vs_volume(all_tasks_scan_timings, all_tasks_volumes, output_dir):
    plt.figure(figsize=(14, 8))
    markers = ['o', 's', 'D', '^', 'v']
    colors = {
        'BrainTumour': 'yellow',
        'Heart': 'red',
        'Colon': 'brown',
        'HepaticVessel': 'purple',
        'Hippocampus': 'pink',
        'Lung': 'green',
        'Pancreas': 'blue',
        'Prostate': 'orange'
    }
    for i, (task, scan_timings) in enumerate(all_tasks_scan_timings.items()):
        if task in ['NIHPancreas', 'Spleen']:
            continue
        volumes = all_tasks_volumes.get(task, {})
        for j, (config, timings) in enumerate(scan_timings.items()):
            if len(timings) == 0 or config not in volumes:
                continue
            median_time = np.median(timings)
            lower_bound = np.percentile(timings, 25)
            upper_bound = np.percentile(timings, 75)
            mean_volume = np.mean([v for _, v in volumes[config]])
            plt.errorbar(mean_volume, median_time, yerr=[[median_time - lower_bound], [upper_bound - median_time]], fmt=markers[j % len(markers)], color=colors[task])
    
    # Create separate legends for shapes and colors
    shape_handles = [plt.Line2D([0], [0], marker=markers[i], color='w', markerfacecolor='k', markersize=10) for i in range(len(markers))]
    shape_labels = ['Default', 'No TTA', 'Flash', 'Fastest no MP', 'No TTA and 1 Fold']
    color_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[task], markersize=10) for task in colors.keys()]
    color_labels = list(colors.keys())
    
    shape_legend = plt.legend(shape_handles, shape_labels, title="Configurations", loc="upper left", bbox_to_anchor=(1, 1))
    plt.gca().add_artist(shape_legend)
    plt.legend(color_handles, color_labels, title="Tasks", loc="upper left", bbox_to_anchor=(1, 0.5))
    
    plt.xlabel('Mean Volume (mm^3)', fontsize=14)
    plt.ylabel('Median Inference Time (seconds)', fontsize=14)
    plt.title('Median Inference Time vs. Volume Across All Tasks', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(os.path.join(output_dir, 'combined_median_time_vs_volume.png'))
    plt.close()

def evaluate_task(task_name, model_type):
    """
    Evaluate the specified task and model type.

    Args:
        task_name (str): The name of the task.
        model_type (str): The model type (2d or 3d).
    """
    inference_base_dir = 'Inference_outputs'
    task_folder_path = os.path.join(inference_base_dir, model_type, task_name)
    labels_base_dir = os.path.join('nnUNet_raw_data_base', 'nnUNet_raw_data', f'Task062_{task_name}', 'labelsTr')
    
    if not os.path.isdir(task_folder_path):
        print(f"Task folder {task_folder_path} does not exist.")
        return
    
    is_nih_pancreas = task_name == "NIHPancreas"

    if not is_nih_pancreas:
        volumes = compare_volumes(task_folder_path)
        stats = compute_statistics(volumes)

        if not stats:
            print(f"No valid volumes found for task {task_name} with model type {model_type}.")
            return

        # Save volume results to a file
        output_file_path = os.path.join(task_folder_path, 'volume_comparison.txt')
        with open(output_file_path, 'w') as f:
            for folder_name, volume_list in volumes.items():
                f.write(f"Volumes for {folder_name}:\n")
                for file_name, volume in volume_list:
                    f.write(f"{file_name}: Volume={volume:.2f} mm^3\n")
                f.write("\n")
            f.write("Summary Statistics:\n")
            for folder_name, stat in stats.items():
                f.write(f"Statistics for {folder_name}:\n")
                f.write(f"Median Volume: {stat['median_volume']:.2f} mm^3\n")
                f.write(f"Standard Deviation: {stat['std_volume']:.2f} mm^3\n")
                f.write(f"Min Volume: {stat['min_volume']:.2f} mm^3\n")
                f.write(f"Max Volume: {stat['max_volume']:.2f} mm^3\n")
                f.write("\n")

        print(f"Volume results saved to {output_file_path}")

        plot_statistics(stats, task_folder_path)

        print(f"Volume figures saved to {task_folder_path}")

    # Compute DICE scores only for NIHPancreas
    if is_nih_pancreas:
        dice_scores = []
        for file_name in os.listdir(task_folder_path):
            if file_name.endswith('.nii.gz'):
                file_path = os.path.join(task_folder_path, file_name)
                segmentation, _ = load_nifti(file_path)
                label_file_path = os.path.join(labels_base_dir, file_name.replace("pancreas_", "label"))
                if os.path.isfile(label_file_path):
                    ground_truth, _ = load_nifti(label_file_path)
                    if segmentation.shape != ground_truth.shape:
                        ground_truth = np.transpose(ground_truth, (2, 1, 0))
                    dice_score = compute_dice_score(segmentation, ground_truth)
                    dice_scores.append((file_name, dice_score))

        # Save DICE score results to a file
        dice_output_file_path = os.path.join(task_folder_path, 'dice_scores.txt')
        with open(dice_output_file_path, 'w') as f:
            f.write("DICE Scores for NIHPancreas:\n")
            for file_name, score in dice_scores:
                f.write(f"{file_name}: DICE Score={score:.4f}\n")
            f.write("\n")

        print(f"DICE score results saved to {dice_output_file_path}")

    # Load scan timings and plot distribution
    scan_timings = load_scan_timings(task_folder_path)
    for config, timings in scan_timings.items():
        if len(timings) > 0:
            plot_scan_timings_boxplot(timings, os.path.join(task_folder_path, config), config)
    plot_inference_times_stacked_lineplot(scan_timings, task_folder_path, f'Inference Times for {task_name}')
    if not is_nih_pancreas:
        plot_median_time_vs_volume(scan_timings, volumes, task_folder_path, f'Median Inference Time vs. Volume for {task_name}')

def main():
    """
    Main function to parse command line arguments and run the evaluation.
    """
    parser = argparse.ArgumentParser(description="Evaluate nnUNet inference results.")
    parser.add_argument("-all", action="store_true", help="Evaluate all tasks for all model types.")
    parser.add_argument("-tasks", nargs='+', help="Specify the task names to evaluate.")
    parser.add_argument("-model_type", choices=["2d", "3d"], required=True, help="Specify the model type to use (2d or 3d).")
    parser.add_argument("-combined_only", action="store_true", help="Compute only the combined median time plot and flash distribution plot.")
    
    args = parser.parse_args()
    
    all_tasks_scan_timings = {}
    all_tasks_volumes = {}

    if args.all or args.combined_only:
        # Load scan timings and volumes for all tasks
        inference_base_dir = 'Inference_outputs'
        for task_folder in os.listdir(os.path.join(inference_base_dir, args.model_type)):
            task_folder_path = os.path.join(inference_base_dir, args.model_type, task_folder)
            if os.path.isdir(task_folder_path):
                print(f"Loading data for {task_folder} for model type {args.model_type}")
                all_tasks_scan_timings[task_folder] = load_scan_timings(task_folder_path)
                all_tasks_volumes[task_folder] = compare_volumes(task_folder_path)
    
    if args.all:
        # Evaluate all tasks for the specified model type
        for task_folder in all_tasks_scan_timings.keys():
            print(f"Evaluating {task_folder} for model type {args.model_type}")
            evaluate_task(task_folder, args.model_type)
    
    if args.combined_only:
        # Plot combined median time vs. volume across all tasks
        plot_combined_median_time_vs_volume(all_tasks_scan_timings, all_tasks_volumes, 'Inference_outputs')

    elif args.tasks:
        # Evaluate the specified tasks for the specified model type
        for task_name in args.tasks:
            print(f"Evaluating {task_name} for model type {args.model_type}")
            evaluate_task(task_name, args.model_type)
            task_folder_path = os.path.join('Inference_outputs', args.model_type, task_name)
            all_tasks_scan_timings[task_name] = load_scan_timings(task_folder_path)
            all_tasks_volumes[task_name] = compare_volumes(task_folder_path)

    if not args.combined_only:
        # Plot combined median time vs. volume across all tasks
        plot_combined_median_time_vs_volume(all_tasks_scan_timings, all_tasks_volumes, 'Inference_outputs')

if __name__ == "__main__":
    main()