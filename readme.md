# Machine Learning Model Evaluation Toolkit

This project provides a comprehensive toolkit for evaluating the performance of machine learning models using various metrics such as AUROC, AUPRC, Brier score, calibration slope, and more. It is designed to be modular, easy to use, and extensible for custom evaluation needs.

## Features

- **Performance Metrics**: Calculate AUROC, AUPRC, Sensitivity, Specificity, Precision, F1 Score, Accuracy, Brier Score, and more.
- **Calibration Metrics**: Evaluate model calibration using Integrated Calibration Index (ICI), calibration slope, intercept, and Hosmer-Lemeshow test.
- **Bootstrap Confidence Intervals**: Compute confidence intervals for metrics using bootstrapping.
- **Modular Design**: Easily extend or modify the toolkit to include additional metrics or evaluation methods.

## Installation

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

Alternatively, you can install the package directly using `setup.py`:

```bash
python setup.py install
```

## Usage

### Running the Evaluation

To evaluate model predictions, modify the `main.py` file to specify your data paths and model predictions, then run:

```bash
python src/main.py
```

This will generate a CSV file (`tmp_result_with_averages.csv`) containing the evaluation results with confidence intervals.

### Example Code

Here's an example of how to use the toolkit in your own script:

```python
from src.data.load_data import load_data
from src.evaluation.evaluate import process_multiple_files

# Load data
train_path = 'path/to/train_data.csv'
valid_path = 'path/to/valid_data.csv'
test_path = 'path/to/test_data.csv'
outcome_var = 'target_column'
input_vars = ['feature1', 'feature2', 'feature3']

(x_train, y_train), (x_valid, y_valid), (x_test, y_test) = load_data(train_path, valid_path, test_path, outcome_var, input_vars)

# Define file paths for predictions
file_paths = [
    {
        'name': 'model1',
        'valid': 'path/to/model1_valid_predictions.pkl',
        'test': 'path/to/model1_test_predictions.pkl'
    },
    {
        'name': 'model2',
        'valid': 'path/to/model2_valid_predictions.pkl',
        'test': 'path/to/model2_test_predictions.pkl'
    },
]

# Define metrics to calculate
metric_names = ['AUROC', 'AUPRC', 'Sensitivity', 'Specificity', 'Precision', 'F1', 'Accuracy', 'Brier', 'ICI', 'Calibration Slope', 'Calibration Intercept', 'Unreliability p-value']

# Process files and calculate metrics
results_df = process_multiple_files(file_paths, metric_names, y_valid, y_test)

# Save results
results_df.to_csv('evaluation_results.csv', index=False)
```

## Project Structure

```
project_name/
│
├── README.md                   # Project documentation
├── requirements.txt            # List of dependencies
├── setup.py                    # Setup script for package installation
├── src/                        # Source code
│   ├── __init__.py             # Package initialization
│   ├── metrics/                # Metrics calculation modules
│   │   ├── __init__.py
│   │   ├── calibration.py      # Calibration-related metrics
│   │   ├── classification.py   # Classification metrics
│   │   └── utils.py            # Utility functions
│   ├── data/                   # Data loading and preprocessing
│   │   ├── __init__.py
│   │   └── load_data.py        # Data loading functions
│   ├── evaluation/             # Evaluation pipeline
│   │   ├── __init__.py
│   │   └── evaluate.py         # Main evaluation functions
│   └── main.py                 # Entry point for the evaluation
└── tests/                      # Unit tests
    ├── __init__.py
    ├── test_metrics.py         # Tests for metrics
    └── test_evaluation.py      # Tests for evaluation pipeline
```