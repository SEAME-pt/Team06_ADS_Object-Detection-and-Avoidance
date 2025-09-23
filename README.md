# YOLO Traffic Sign Detection Architecture

This repository contains a complete architecture for creating, processing, and training YOLO (You Only Look Once) models to detect traffic signs. Built with a combination of Unity for dataset generation and Python scripts for dataset management and model training, this project streamlines the process of developing accurate object detection models for specific traffic sign categories.

## Overview

The architecture is designed to support end-to-end development of YOLO-based object detection systems, with a focus on traffic sign recognition. It includes tools for generating synthetic datasets in Unity, splitting datasets into training and validation sets, analyzing dataset balance, visualizing annotations, and training YOLO models with customized augmentation and hyperparameters.

Key components include:
- **Unity Dataset Generator**: A custom Unity tool to create synthetic images of traffic signs with corresponding YOLO-format annotations.
- **Dataset Splitter**: A Python script (`createDataSet.py`) to organize and split datasets into training and validation subsets with stratified sampling.
- **Dataset Analyzer**: A Python script (`margins.py`) to evaluate dataset balance and suggest data augmentation strategies.
- **Label Visualizer**: A Pygame-based tool (`showlabels.py`) to visualize bounding boxes and validate annotations.
- **YOLO Training Script**: A Python script (`train.py`) to train YOLO models with tailored configurations and augmentation settings.

This project is ideal for computer vision researchers and developers working on traffic sign detection, offering a modular and customizable pipeline.

## Features

- **Synthetic Dataset Creation**: Generate diverse traffic sign images using Unity with automated annotation in YOLO format.
- **Stratified Dataset Splitting**: Ensure balanced distribution of classes across training (80%) and validation (20%) sets.
- **Dataset Analysis**: Assess class distribution and balance, with visualizations and suggestions for data augmentation.
- **Annotation Visualization**: Preview bounding boxes on images to validate dataset quality before training.
- **Custom YOLO Training**: Train YOLO models (e.g., YOLOv8 or YOLOv11) with configurable hyperparameters and augmentation techniques.
- **Traffic Sign Focus**: Supports 10 specific traffic sign classes, including STOP, SPEED_50, and SEMAFORO_VERDE.

## Repository Structure

- **Unity Tool**: Scripts and assets for generating synthetic traffic sign datasets (referenced in project context).
- **Dataset Processing**:
  - `createDataSet.py`: Splits datasets into training and validation sets with detailed statistics and visualizations.
  - `margins.py`: Analyzes dataset balance and provides augmentation recommendations.
  - `showlabels.py`: Visualizes bounding boxes and generates dataset validation reports.
- **Training**:
  - `train.py`: Configures and trains YOLO models with custom augmentation and hyperparameters.
- **Configuration**:
  - `dataset.yaml`: Defines dataset paths and class names for YOLO training.

## Requirements

- **Unity**: For generating synthetic datasets of traffic signs (version compatible with the dataset generator tool).
- **Python 3.x**: For dataset processing, analysis, and training scripts.
- **Libraries**: Install dependencies such as `ultralytics`, `matplotlib`, `numpy`, `pygame`, `seaborn`, and `pyyaml` using `pip`.
- **YOLO Models**: Pre-trained weights like `yolov8s.pt` or `yolo11n.pt` for training.
- **Hardware**: GPU recommended for faster training (configured for device 0 in scripts).

- The script uses configurations like 200 epochs, image size of 640, and specific augmentation parameters (e.g., HSV adjustments, rotation, mosaic).
- Results are saved under `traffic_signs/yolov8_experiment` with periodic checkpoints.

## Traffic Sign Classes

The architecture supports detection of the following 10 traffic sign classes as defined in `dataset.yaml`:
- STOP
- PASSAGEM
- VEL_50
- VEL_80
- SEMAFORO_VERMELHO
- SEMAFORO_VERDE
- SEMAFORO_LARANJA
- PASSADEIRA
- DANGER
- CURVA

 
 
## Acknowledgments

This architecture leverages the power of YOLO for real-time object detection, combined with Unity for synthetic data generation, to address challenges in traffic sign recognition. It builds on open-source tools and libraries like Ultralytics YOLO, Pygame, and Matplotlib for a robust pipeline.
