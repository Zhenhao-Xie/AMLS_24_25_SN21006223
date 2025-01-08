# AMLS_24_25_SN21006223

This repository contains two classification tasks (**Task A** and **Task B**) using three different machine learning models: **RandomForest (RF)**, **SVM**, and **ResNet-18**. The code for each task is structured in a self-contained manner, with model training procedures encapsulated in **BreastClassification** (Task A) and **BloodClassification** (Task B).

## Project Structure

```
AMLS_24_25_SN21006223
├── A
│   ├── breast_classification.py
│   ├── confusion_matrices_breast.png
│   └── resnet_loss_curve_breast.png
├── B
│   ├── blood_classification.py
│   ├── confusion_matrices_blood.png
│   └── resnet_loss_curve_blood.png
├── Datasets
│   ├── breastmnist.npz
│   └── bloodmnist.npz
├── main.py
└── README.md
```

### Role of Each File

- **A/breast_classification.py**  
  Contains the `BreastClassification` class for Task A (BreastMNIST).  
  Encapsulates code for training RF, SVM, and ResNet-18, including hyperparameter searching.

- **A/confusion_matrices_breast.png**  
  Confusion matrix plot for the three models on the BreastMNIST dataset.

- **A/resnet_loss_curve_breast.png**  
  Training loss curves for the best ResNet-18 model (BreastMNIST).

- **B/blood_classification.py**  
  Contains the `BloodClassification` class for Task B (BloodMNIST).  
  Encapsulates code for training RF, SVM, and ResNet-18, including hyperparameter searching.

- **B/confusion_matrices_blood.png**  
  Confusion matrix plot for the three models on the BloodMNIST dataset.

- **B/resnet_loss_curve_blood.png**  
  Training loss curves for the best ResNet-18 model (BloodMNIST).

- **Datasets/**  
  Stores the `.npz` files (`breastmnist.npz`, `bloodmnist.npz`) automatically downloaded by `medmnist`.  
  Once downloaded, the dataset files will be reused in subsequent runs.

- **main.py**  
  Illustrates how to initialize the two classification tasks (`BreastClassification` and `BloodClassification`), train the models (RF, SVM, ResNet-18), and visualize the results.  

- **README.md**  
  The current file, providing an overview of the project, instructions, and dependencies.

## Dependencies

The code has been tested with the following package versions:
- **torch** 2.5.0
- **torchvision** 0.20.0
- **scikit-learn** (sklearn) 1.5.1
- **numpy** 1.26.4
- **matplotlib** 3.8.4
- **medmnist** 3.0.2

## Usage Instructions

1. **Ensure Dependencies**  
   Make sure the required libraries are installed before running any code.

2. **Project Organization**  
   - Make sure your terminal’s working directory is set to `AMLS_24_25_SN21006223`.
   - If `medmnist` is correctly installed, the *BreastMNIST* and *BloodMNIST* data will be automatically downloaded to the `Datasets/` folder upon first run of `main.py`.

3. **Run the Code**  
   - Execute `python main.py`.  
   - The script will:
     - Instantiate and run both `BreastClassification` (Task A) and `BloodClassification` (Task B).
     - For each task, it will train **RandomForest**, **SVM**, and **ResNet-18** with user-defined hyperparameter grids (when applicable).
     - Generate confusion matrices and (for ResNet-18) training curves.

4. **Hyperparameter Search**  
   - Both `breast_classification.py` and `blood_classification.py` support user-defined hyperparameter grids for **RF**, **SVM**, and **ResNet-18**.  
   - Since **BloodMNIST** is relatively larger, searching through too many hyperparameter candidates may take a long time. Adjust accordingly.

5. **GPU/CPU**  
   - If CUDA is detected, the code will use the GPU for training ResNet-18; otherwise, it will train on CPU.