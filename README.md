# 3D Medical Image Segmentation with VNet

## Overview

This project focuses on 3D medical image segmentation using the VNet architecture. The primary goal is to segment multiple organs from CT scans, including the liver, right kidney, left kidney, and spleen. The project involves preprocessing CT images, resampling and normalizing them, training a 3D segmentation model, and evaluating its performance.

## Setup Instructions

To set up the environment and run the code, follow these steps:

1. **Clone the Repository** (if applicable):

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Create a Virtual Environment**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   Make sure you have `pip` installed and then run:

   ```bash
   pip install -r requirements.txt
   ```

   The `requirements.txt` file should include:

   ```
   nibabel
   numpy
   pandas
   matplotlib
   SimpleITK
   torch
   scikit-learn
   ```

4. **Prepare the Data**:

   - Place your raw CT images and label files in the `images_path` and `labels_path` directories respectively.
   - Ensure the data is organized in the format expected by the code.

5. **Run the Jupyter Notebook**:
   Launch Jupyter Notebook and open the notebook file. Execute each cell sequentially to preprocess the data, train the model, and evaluate the results.

## Model Architecture

The chosen model for this project is the **VNet** architecture, a 3D U-Net variant optimized for medical image segmentation.

### Key Architectural Details:

- **Encoder**: The downsampling path consists of four convolutional blocks with increasing feature channels: 16, 32, 64, and 128.
- **Bottleneck**: A convolutional block with 256 channels.
- **Decoder**: The upsampling path includes transposed convolutions and concatenation with corresponding encoder layers, reducing channels back to 16.
- **Output Layer**: A final 3D convolution layer with kernel size 1 to produce segmentation maps for each organ class.

## Training Process

### Data Preprocessing:

1. **Resampling**: CT images and labels are resampled to a uniform size of `(128, 128, 128)` using SimpleITK.
2. **Normalization**: CT scans are normalized to the HU range of `[-1000, 400]` and scaled to `[0, 1]`.
3. **Dataset Splitting**: Data is split into training (70%), validation (15%), and test (15%) sets.

### Training Procedure:

- **Loss Functions**: Combination of Cross-Entropy Loss and Dice Loss.
- **Optimizer**: Adam optimizer with an initial learning rate of `1e-4`.
- **Scheduler**: Reduce learning rate on plateau based on validation loss.
- **Epochs**: The model is trained for 50 epochs.

## Validation and Inference

### Validation:

- The validation process evaluates the model on unseen data and computes the Dice Score to assess the overlap between predicted and ground truth labels.
- **Dice Score**: The Dice Score is calculated for each organ to measure the model's performance in segmenting each organ.

### Inference:

- The model is used to predict organ segments on test images. The results are saved for further analysis.

## 3D Visualization

For visualizing the 3D rendered segments of the predicted organs, you can use specialized software or scripts to create a video demonstration. Ensure to capture the segmented organs in 3D and provide visual comparisons with the ground truth.

The 3D demonstration video of the segmentation model would be uploaded soon.
