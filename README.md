 Linear Regression GUI — From Scratch

A desktop application developed with Python and Tkinter for building and evaluating a Linear Regression model from CSV datasets.
The model is implemented entirely from scratch without relying on machine learning libraries for the training process.

 Features

* Linear Regression implemented from scratch
* Supports both Batch Gradient Descent and Stochastic Gradient Descent (SGD)
* Manual train/test data splitting
* Feature standardization before training
* L2 Regularization to reduce overfitting
* Live visualization of the training loss
* Model evaluation using MSE, RMSE, and R² Score

 Requirements

```bash
pip install numpy pandas matplotlib scikit-learn
```

 Run the Project

```bash
python linear_gui_with_split.py
```

 Usage

1. Load a CSV dataset.
2. Select the target column.
3. Choose the optimization method (Batch GD or SGD).
4. Configure the training parameters.
5. Start training and monitor the loss curve along with the evaluation metrics.

 Author

Mohamed Ryad
Faculty of Artificial Intelligence — IoT Department
