# Machine Learning & Data Science Internship

**Arafat Rahman** · Datheon · Application ID: `DT-2026-JTNGO`

Four completed projects covering multiclass classification, housing regression, customer churn and review sentiment. Each project includes an executed Jupyter notebook, documented methods, saved results and evaluation figures.

[Official internship requirements](https://www.datheon.in/training/catalog/machine-learning-and-data-science-internship)

## Projects

| Project | Documentation | Notebook | Kaggle | Status |
|:--|:--|:--|:--|:--|
| Iris Classification | [Overview](Task_01_Classification/README.md) | [Open notebook](Task_01_Classification/iris_classification.ipynb) | [View](https://www.kaggle.com/code/arafat1746/iris-classification) | Completed |
| Housing Price Prediction | [Overview](Task_02_Housing_Price_Prediction/README.md) | [Open notebook](Task_02_Housing_Price_Prediction/housing_price_prediction.ipynb) | [View](https://www.kaggle.com/code/arafat1746/housing-price-prediction) | Completed |
| Customer Churn Classification | [Overview](Task_03_Customer_Churn_Classification/README.md) | [Open notebook](Task_03_Customer_Churn_Classification/customer_churn_classification.ipynb) | [View](https://www.kaggle.com/code/arafat1746/customer-churn-classification) | Completed |
| Sentiment Classification | [Overview](Task_04_Sentiment_Classification/README.md) | [Open notebook](Task_04_Sentiment_Classification/sentiment_classification.ipynb) | [View](https://www.kaggle.com/code/arafat1746/sentiment-classification) | Completed |

## Results at a glance

Results below come from the saved local executions. Scores compare models within each project; the datasets and metrics differ between projects.

| Project | Model | Held-out result |
|:--|:--|:--|
| Iris Classification | Logistic Regression / Decision Tree | Both: 93.33% accuracy and 0.9333 macro F1 |
| Housing Price Prediction | OLS | RMSE $68,845.00; MAE $49,458.67; R² 0.6383 |
| Housing Price Prediction | Ridge | RMSE $68,897.95; MAE $49,469.16; R² 0.6378 |
| Housing Price Prediction | Lasso | RMSE $68,937.05; MAE $49,426.84; R² 0.6373 |
| Customer Churn Classification | Random Forest | ROC-AUC 0.8350; churn recall 0.7406 |
| Customer Churn Classification | XGBoost | ROC-AUC 0.8463; churn recall 0.7861 |
| Sentiment Classification | Multinomial Naive Bayes | Accuracy 80.34%; macro F1 0.8033 |
| Sentiment Classification | Logistic Regression | Accuracy 82.22%; macro F1 0.8222 |

- **Iris:** both models made the same two errors on 30 test examples.
- **Housing:** training cross-validation selected Lasso; OLS had a slightly lower test RMSE. The test set was not used to change the selected model.
- **Churn:** training cross-validation selected XGBoost. It detected more churners but also produced more false positives. Kaggle results differ slightly and are documented separately in the project README.
- **Sentiment:** Logistic Regression had higher observed accuracy and macro F1 on this split. Local and Kaggle test predictions match.

## Datasets and evaluation

| Project | Dataset | Training rows | Test rows |
|:--|:--|--:|--:|
| Iris Classification | [Seaborn Iris CSV](https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv) | 120 | 30 |
| Housing Price Prediction | [California housing CSV](https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv) | 16,512 | 4,128 |
| Customer Churn Classification | [IBM Telco Customer Churn](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv) | 5,634 | 1,409 |
| Sentiment Classification | [UCI Sentiment Labelled Sentences](https://doi.org/10.24432/C57604) | 2,340 | 585 |

**Sentiment dataset substitution:** the official CSV returned HTTP 404 on 25 October 2026. The notebook uses the attributed UCI dataset instead. Cleaning removes two empty texts and 73 normalized duplicates from 3,000 source sentences, leaving 2,925 rows.

All projects use an 80/20 split with `random_state=42`; classification splits are stratified. Learned preprocessing is fitted on training data, and inside training folds where cross-validation is used. Project documentation records duplicate handling, source fingerprints, evaluation details and limitations. The churn split contains 12 shared predictor profiles belonging to different customer IDs; its README discusses the resulting limitation.

## Repository structure

```text
Machine_Learning_Data_Science_Internship/
├── Task_01_Classification/
│   ├── iris_classification.ipynb
│   ├── README.md
│   ├── results/
│   └── figures/
├── Task_02_Housing_Price_Prediction/
│   ├── housing_price_prediction.ipynb
│   ├── README.md
│   ├── results/
│   └── figures/
├── Task_03_Customer_Churn_Classification/
│   ├── customer_churn_classification.ipynb
│   ├── README.md
│   ├── results/
│   └── figures/
├── Task_04_Sentiment_Classification/
│   ├── sentiment_classification.ipynb
│   ├── README.md
│   ├── results/
│   └── figures/
├── README.md
├── requirements.txt
└── .gitignore
```

Notebooks stay in their project folders. `results/` contains calculated tables and reports; `figures/` contains generated PNG charts. Virtual environments, credentials and raw dataset downloads are excluded from Git.

## Setup and execution

Tested with **Python 3.12.10 on Windows**, VS Code and Microsoft's Jupyter extension. The shared [requirements.txt](requirements.txt) pins NumPy, Pandas, SciPy, Scikit-learn, Matplotlib, Seaborn, XGBoost, NLTK and Jupyter dependencies.

From the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m ipykernel install --user --name internship-projects --display-name "Python (Internship Projects)"
```

Open any notebook in VS Code, choose **Python (Internship Projects)** and select **Run All**. Directly invoking the virtual environment's Python avoids PowerShell activation-script restrictions.

For command-line execution, choose a notebook path from the project table. For example:

```powershell
.\.venv\Scripts\python.exe -m jupyter nbconvert --execute --to notebook --inplace Task_04_Sentiment_Classification/sentiment_classification.ipynb --ExecutePreprocessor.kernel_name=internship-projects --ExecutePreprocessor.timeout=600
```

Internet access is needed to download the datasets. The sentiment notebook also downloads NLTK stopwords and WordNet resources. Each project's README includes its own kernel setup and execution command. Rerunning a notebook regenerates its saved results and figures.

## Verification

| Project | Executed code cells | Saved figures | Result files |
|:--|--:|--:|--:|
| Iris Classification | 19 | 6 | 2 |
| Housing Price Prediction | 19 | 7 | 6 |
| Customer Churn Classification | 17 | 6 | 8 |
| Sentiment Classification | 14 | 4 | 7 |

Each notebook completed two fresh local kernel runs with saved outputs and reproducibility checks. Metrics were checked against predictions or confusion-matrix counts, and saved figures were visually inspected. Kaggle execution and GitHub publication were also verified. Full notebook browser-page inspection was not performed for every project; detailed verification notes remain in the project READMEs.

## Progress

- [x] Iris Classification — implemented, executed and published.
- [x] Housing Price Prediction — implemented, executed and published.
- [x] Customer Churn Classification — implemented, executed and published.
- [x] Sentiment Classification — implemented, executed and published.

All four technical projects are complete. LinkedIn publication is not recorded as completed here, and the final internship form has not been submitted. Project completion does not imply Datheon approval or certificate issuance.
