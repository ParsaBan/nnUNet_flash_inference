# nnU-Net Flash Inference

_Flash_ is one of five inference configurations that improve the efficiency of nnU-Net Inference. This project aims to improve the speed at which inference is conducted, with minimal side effects on segmentation accuracy and quality. _Flash_ can speed up inference by up to 25x for certain datasets. Larger datasets benefit more from _Flash_ than smaller datasets, however, the accuracy tradeoffs are also much more noticeable with larger datasets.

![image](https://github.com/user-attachments/assets/79d7e889-6577-4fd1-9533-bf0b45f77f09)
<br><br><be>
<div align="center">
  
| **Configuration**      | **Time (s)** | **Median time/scan (s)** |
|------------------------|-------------|--------------------------|
| Default                | 16,585.5    | 99.21                    |
| No TTA                 | 2,926.0     | 18.5                     |
| No TTA & 1 Fold        | 642.3       | 4.1                      |
| Flash                  | 657.7       | 3.9                      |
| Flash No MP            | 938.5       | 5.8                      |

*Aggregate and per scan time taken for the 3D pancreas dataset in all five configurations (n=138).*
</div>

## Preamble
_Flash Inference_ is currently exclusively operable on the ten datasets from MSD 2018, which include:
* Brain tumours
* Heart
* Liver
* Hippocampus
* Prostate
* Lung
* Pancreas
* HepaticVessel
* Spleen
* Colon

## Installation
Please refer to the **nnU-Net V1** installation guide, provided here: https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1

Please download any, or all, of the ten datasets from the MSD:
http://medicaldecathlon.com/dataaws/

Alternatively, you may download the pre-trained models directly from nnU-Net by running the following command:
```nnUNet_download_pretrained_model Task0XX_Y```

For example:
```nnUNet_download_pretrained_model Task005_Prostate```

## Data Conversion
By default, the Decathlon data comes as 4D niftis, incompatible with nnU-Net. For each dataset, run the following command:
```nnUNet_convert_decathlon_task -i /xxx/TaskXX_Y```

For example:
```nnUNet_convert_decathlon_task -i /xxx/Task04_Hippocampus```

_Note_: Each dataset has three 'imagesTr', 'labelsTr', 'imagesTs' subfolders! The converted dataset can be found in $nnUNet_raw_data_base/nnUNet_raw_data ($nnUNet_raw_data_base is the folder for raw data that you specified during installation). If you encounter an error with the specificed folders not being found, please refer to the instructions and set the correct environmental variables.

## Usage
After activating your virtual environment, you are ready to begin accelerated inference. In the root directory of the repository, enter ```python accelerated_inference.py -h``` to see all available commands. Below are some example commands you can use based on your use case.
_Note:_ ```model_type``` must be specified each time ```accelerated_inference.py``` is run. The available options are 2D and 3D (nnU-Net's full-resolution 3D U-Net).


Running all five inference configurations on all available datasets in 3D:
```python accelerated_inference.py -model_type 3d -all```

Running _Flash_ on the Pancreas dataset in 3D:
```python accelerated_inference.py -model_type 3d -tasks Pancreas -configs flash```

Running _Flash_ and the default nnU-Net inference configuration on the Heart dataset in 2D with a custom input directory:
```python accelerated_inference.py -model_type 2D -tasks Heart -configs default flash -input_dir /xxx/Task002_Heart/imagesTs```

## Evaluation
Result evaluation is available via the ```evaluate.py``` script. In the root directory of the repository, enter ```python evaluate.py -h``` to see all available commands. Below are some example commands you can use based on your use case. _Note:_ ```model_type``` must be specified each time ```evaluate.py``` is run. The available options are 2D and 3D (nnU-Net's full-resolution 3D U-Net).

Evaluate all available datasets in 3D:
```python evaluate.py -model_type 3d -all```

Evaluate the Pancreas dataset in 2D:
```python evaluate.py -model_type 3d -tasks Pancreas```

Evaluate the Heart, Hippocampus, and Brain Tumour dataset in 3D:
```python evaluate.py -model_type 3d -tasks Heart Hippocampus BrainTumour```

## Warnings:
As stated in nnU-Net's repository, a GPU with at least 4GB of VRAM is required to effectively use Inference. To get desirable results, use a GPU with at least 16GB of VRAM, and have >32GB of system memory. Faster storage (SSDs) will also improve inference times.

