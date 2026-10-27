# Iris Classification

**Arafat Rahman** · Machine Learning & Data Science Internship

[Notebook](iris_classification.ipynb) · [Kaggle](https://www.kaggle.com/code/arafat1746/iris-classification) · [All projects](../README.md)

Compare Logistic Regression and Decision Tree classifiers on four Iris measurements using a stratified holdout, training-only scaling and per-class evaluation.

## Results

Four-feature models evaluated on the same **30 held-out samples**:

| Model | Accuracy | Macro precision | Macro recall | Macro F1 |
|:--|--:|--:|--:|--:|
| Logistic Regression | 93.33% | 93.33% | 93.33% | 93.33% |
| Decision Tree | 93.33% | 93.33% | 93.33% | 93.33% |

Both models correctly classified **28 of 30** test examples and made the same two errors. There is no observed performance advantage on this split. Setosa achieved precision, recall and F1 of 1.00; versicolor and virginica each achieved 0.90, with support of 10 per species.

![Model comparison](figures/model_comparison.png)

## Dataset

Source: [Iris CSV linked by Datheon](https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv), hosted in the Seaborn data repository.

- 150 samples, four numeric features measured in centimetres and one target column.
- Features: sepal length, sepal width, petal length and petal width.
- Species: setosa, versicolor and virginica; 50 samples each.
- No missing values. One repeated row beyond its first occurrence is retained to preserve the benchmark.
- No identical feature vectors occur across training and test partitions in this split.
- The notebook downloads the CSV into memory and prints its SHA-256 fingerprint. No fallback dataset was used.

## Methodology

1. Validate the dataset shape, column names, numeric types, finite values, class labels, missingness and duplicates.
2. Summarize measurements and visualize class balance and pairwise feature distributions.
3. Use `train_test_split(test_size=0.20, stratify=y, random_state=42)`: 120 training samples and 30 test samples, with 40/10 samples per species.
4. Fit `StandardScaler` only on training data. Transform the training and test features with that fitted scaler for Logistic Regression.
5. Train `LogisticRegression(max_iter=1000, random_state=42)` and `DecisionTreeClassifier(random_state=42)`. The tree uses unscaled features because threshold splits do not require standardization.
6. Calculate accuracy, macro precision, macro recall, macro F1, per-class reports and confusion matrices. Macro metrics weight each species equally.
7. Inspect every test error. Separate two-feature models illustrate decision boundaries using petal length and width; these do not generate the reported four-feature metrics.

No model tuning or feature selection is performed using held-out results. EDA is descriptive; model settings and features are fixed in advance.

## Environment and installation

Tested with **Python 3.12.10 on Windows**. Dependencies are pinned in [requirements.txt](../requirements.txt): NumPy, Pandas, SciPy, Scikit-learn, Matplotlib, Seaborn and the Jupyter execution tools.

Run these PowerShell commands from the repository root on a new machine:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m ipykernel install --user --name datheon-task1 --display-name "Python (Datheon Task 1)"
```

Activation is unnecessary: invoking the environment's Python directly avoids PowerShell activation-script restrictions. The `.venv` folder is ignored by Git.

## Run the notebook

In VS Code, open `Task_01_Classification/iris_classification.ipynb`, select **Python (Datheon Task 1)** and choose **Run All**.

Alternatively, launch Jupyter:

```powershell
.\.venv\Scripts\python.exe -m jupyter lab Task_01_Classification/iris_classification.ipynb
```

Execute and save all outputs from PowerShell:

```powershell
.\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace Task_01_Classification/iris_classification.ipynb --ExecutePreprocessor.timeout=180
```

The notebook supports the repository root and the Task 1 folder as working directories. Internet access is required to download the source CSV. Rerunning regenerates the eight results and figure files.

## Figures and saved results

| File | Purpose |
|:--|:--|
| `iris_pairplot.png` | Feature histograms and pairwise relationships, grouped by species |
| `class_distribution.png` | Samples per class |
| `confusion_matrix_logistic_regression.png` | Four-feature Logistic Regression test predictions |
| `confusion_matrix_decision_tree.png` | Four-feature Decision Tree test predictions |
| `model_comparison.png` | Common-scale accuracy and macro metrics |
| `decision_boundaries.png` | Separate two-feature illustration models with training points only |
| `model_comparison.csv` | Full-precision calculated aggregate metrics |
| `classification_reports.txt` | Per-class metrics, macro/weighted averages and support |

## Error analysis and limitations

Both classifiers misclassified source row 77 (versicolor → virginica) and row 134 (virginica → versicolor), using zero-based CSV indices. The notebook shows their feature values. These errors do not establish a causal explanation or justify tuning against the test set.

- The balanced benchmark is small and does not represent arbitrary real-world populations.
- A single 30-sample test split does not measure variability across splits. One changed prediction alters accuracy by about 3.33 percentage points.
- Repeated source measurements are retained, with an explicit train/test overlap check.
- Boundary plots use two features and must not be read as visualizations of the four-feature evaluation models.
- The upstream CSV is branch-based and could change; the saved notebook records the exact source fingerprint.

## Verification

The notebook completed **two sequential runs in separate fresh kernels**, including execution from both supported working directories. All 19 code cells passed. The eight generated artifacts were byte-identical across runs. Metrics were independently checked against confusion-matrix counts, and all six saved figures were visually inspected. The full notebook HTML was exported, but a browser preview was unavailable in the execution environment.
