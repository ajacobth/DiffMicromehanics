# Surrogate Model for DiffMicromehanics

This repository contains a surrogate model for DiffMicromehanics. The surrogate model is designed to predict the behavior of composite materials using a neural network. The repository includes scripts for data generation, model training, evaluation, and a graphical user interface (GUI) for predictions.

## Contents

- `NN_surrogate/`: Contains the neural network surrogate model.
- `create_data.py`: Script to generate data for training and testing the model.
- `data.csv`: Dataset used for training and testing.
- `eval.py`: Evaluation script for the model.
- `gui_predict.py`: GUI script for making predictions using the trained model.
- `main.py`: Main script to run training, evaluation, or the GUI.
- `model_defs.py`: Defines the neural network model architecture.
- `normalization_stats_npz`: Normalization statistics for the dataset.

## Running the Code

To run the code, use the `main.py` script with the appropriate mode:

- **Training:** `python main.py --mode train` to train the model.
- **Evaluation:** `python main.py --mode eval` to evaluate the model.
- **GUI:** `python main.py --mode gui` to launch the GUI for predictions.

## GUI Execution

The GUI allows users to input parameters and receive predictions from the trained model. To use the GUI, run the `main.py` script with the `gui` mode or execute `gui_predict.py` directly.

## Model Evaluation

The `eval.py` script restores the trained model from a checkpoint, loads test data, and evaluates the model's performance. It also applies normalization statistics to the data for accurate evaluation.