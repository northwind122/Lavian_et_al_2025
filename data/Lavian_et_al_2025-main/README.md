# Lavian_et_al_2025
Analysis of zebrafish calcium imaging data for representation of visual information for navigation

This repository contains code for analyzing calcium imaging data recorded from zebrafish larvae to investigate how different types of visual information are represented in the fish heading direction system. This code accompanies the paper "The representation of visual motion and landmark position aligns with heading direction in the zebrafish interpeduncular nucleus" by Lavian et al 2025 (https://www.biorxiv.org/content/10.1101/2024.09.25.614953v2).

## Requirements

- Python 3.7
- Required packages are listed in the environment.yml file

The code provided in this package was tested on Windows 10 operating system. 

All software dependencies (including version numbers) can be found in the environment.yml file in this repository. The suite2p_env.yml file can be used to generate the environment used for suite2p pre-processing of the data. 

Typical install time on a normal desktop computer: less than 1 hour, depending on the number of packages already installed.

Expected run time on a normal desktop coputer: varies heavily between analysis types. Some notebooks will provide output in several minutes while other will provide output in several hours, depending on data type (e.g., whole brain datasets Vs single region datasets) or analysis type (e.g., pixel-wise analysis Vs ROI-wise analysis). 

## Installation

1. Clone or download this repository
2. Create a conda environment using the provided environment file:
conda env create -f environment.yml

3. Activate the environment:
conda activate vis_nav

4. Install the package by navigating to the cloned folder and running:
pip install -e lavian_et_al_2025


## Repository Structure

This repository contains Jupyter notebooks organized into the following categories:

- `preprocessing/`: Notebooks for processing raw calcium imaging data
- `visual_motion/`: Analysis of the representation of visual motion in te zebrafish larvae brain
- `landmarks/`: Analysis of the representation of landmark information in the abenula and IPN
- `heading_direction/`: Analysis of representation of visual motion in HD neurons
- `ablations/`: Analysis of the effects of targeted habenuka ablations on the head direction system and representation of visual information
- `anatomy/`: A notebook for generation of anatomy plots
- `morphing/`: notebooks for registration of different datasets to a reference brain. 

## Usage

The analysis is organized into Jupyter notebooks. To run the notebooks:

1. Download the paper data and enter data location in the lavian_et_al_2025/data_location.py script. The variable "master_all" should be the path to the all datasets. The rest should not be edited. 

2. Ensure you have activated the conda environment:
conda activate vis_nav

3. Launch Jupyter:
jupyter lab

4. Navigate to the desired analysis folder and open the relevant notebook. For example, to generate the plots presented in Figure 1C of the paper, go to the folder "visual motion" and run notebook "Fig 1c_d_e tuning to visual motion.ipynb".

Note: all code used for analysis in this paper is provided. However, notebooks provided in the pre-processing and morphing folders will not run without raw data files which are not provided due to their large size. All other notebooks could be run using the provided processed data. 

## License
This project is licensed under the MIT License - see the LICENSE file for details.


## Citation

If you use this code in your research, please cite:

The representation of visual motion and landmark position aligns with heading direction in the zebrafish interpeduncular nucleus
Hagar Lavian, Ot Prat, Luigi Petrucco, Vilim Štih, Ruben Portugues
bioRxiv 2024.09.25.614953; doi: https://doi.org/10.1101/2024.09.25.614953




