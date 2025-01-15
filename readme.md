# Bootstrap Metrics

This repository provides a comprehensive toolkit for calculating and evaluating performance metrics using bootstrapping techniques. It is designed to assess the performance of machine learning models, particularly in clinical and healthcare settings, where robust evaluation is critical.

---

## Features

- **Performance Metrics**: Calculate a wide range of performance metrics, including:
  - AUROC (Area Under the Receiver Operating Characteristic Curve)
  - AUPRC (Area Under the Precision-Recall Curve)
  - Sensitivity, Specificity, Precision, F1 Score, Accuracy
  - Brier Score, Integrated Calibration Index (ICI), Calibration Slope, Calibration Intercept
  - Hosmer-Lemeshow Test p-value

- **Bootstrapping**: Use bootstrapping to estimate confidence intervals for performance metrics, ensuring robust evaluation.

- **Flexible Input**: Supports multiple file formats (e.g., `.pkl`, `.npy`) for predictions and labels.

- **Automated Workflow**: Automatically processes multiple prediction files and generates a consolidated results table.

---

## Repository Structure

```
bootstrap_metrics/
├── src/                   # Source code
│   ├── config.py          # Configuration settings (e.g., file paths, metric names)
│   ├── metrics.py         # Functions for calculating performance metrics
│   ├── utils.py           # Utility functions (e.g., data loading, file processing)
│   └── main.py            # Main script to run the pipeline
├── data/                  # Data files (e.g., train.csv, valid.csv, test.csv)
├── models/                # Model predictions (e.g., .pkl, .npy files)
├── results/               # Output results (e.g., results.csv)
├── README.md              # This file
└── requirements.txt       # Python dependencies
```

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/bootstrap_metrics.git
   cd bootstrap_metrics
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up the data and prediction files:
   - Place your data files (`train.csv`, `valid.csv`, `test.csv`) in the `data/` directory.
   - Place your prediction files (e.g., `.pkl`, `.npy`) in the `models/predictions/` directory.

---

## Usage

To run the pipeline, execute the following command:

```bash
python src/main.py
```

This will:
1. Load the data and predictions.
2. Calculate performance metrics using bootstrapping.
3. Generate a consolidated results table (`results/results.csv`).

---

## Configuration

Modify the `src/config.py` file to customize the following:
- **Data Paths**: Update `TRAIN_DATA_PATH`, `VALID_DATA_PATH`, and `TEST_DATA_PATH` to point to your data files.
- **Prediction Paths**: Update `PRED_DIR` to point to your prediction files.
- **Metrics**: Add or remove metrics from the `METRIC_NAMES` list as needed.

---

## Example Output

The output is saved in `results/results.csv` and includes the following columns:
- **File**: Name of the prediction file.
- **AUROC**: Area Under the ROC Curve with 95% confidence interval.
- **AUPRC**: Area Under the Precision-Recall Curve with 95% confidence interval.
- **Sensitivity**, **Specificity**, **Precision**, **F1**, **Accuracy**: Classification metrics with 95% confidence intervals.
- **Brier**, **ICI**, **Calibration Slope**, **Calibration Intercept**, **Unreliability p-value**: Calibration metrics with 95% confidence intervals.

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---