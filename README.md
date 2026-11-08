# Shopper Purchase Intent Classifier

Classification of online shopping sessions as purchase / no purchase using five machine learning models, with an interactive Streamlit app for evaluating each saved model on an uploaded test CSV.

---

## a. Problem statement

An online store wants to know which browsing sessions end in a purchase. Given session-level behaviour (pages viewed, time spent, bounce and exit rates, page value) and visitor information (month, traffic type, visitor type, and so on), the task is to predict the target `Revenue`:

- `True`: the session ended with a purchase
- `False`: the session ended without a purchase

This is a **binary classification** problem. The positive class (a purchase) is the minority class, so accuracy alone is not a reliable measure and the evaluation also uses AUC, precision, recall, F1 and MCC.

## b. Dataset description

| Item | Detail |
|---|---|
| Name | Online Shoppers Purchasing Intention Dataset |
| Source | UCI Machine Learning Repository (dataset ID 468) |
| Link | https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset |
| Instances | 12,330 sessions |
| Input features | 17 (10 numeric, 7 categorical) |
| Target | `Revenue` (binary) |
| Class balance | Imbalanced: roughly 85% no purchase, 15% purchase |

**Numeric features:** `Administrative`, `Administrative_Duration`, `Informational`, `Informational_Duration`, `ProductRelated`, `ProductRelated_Duration`, `BounceRates`, `ExitRates`, `PageValues`, `SpecialDay`

**Categorical features:** `Month`, `OperatingSystems`, `Browser`, `Region`, `TrafficType`, `VisitorType`, `Weekend`

The dataset meets the assignment minimums (at least 12 features and 500 instances).

## c. GitHub repository link

https://github.com/brindhamadhu27-max/shopper-purchase-intent-classifier

**Live Streamlit app:** `PASTE_YOUR_DEPLOYED_APP_URL`

## d. Models used and evaluation

All five models were trained on the same data, with the same train/test split:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors
4. Gaussian Naive Bayes
5. Support Vector Machine (RBF kernel, with probability estimates)

Metrics were computed on a stratified 20% hold-out test set (about 2,466 rows). Precision, recall and F1 are calculated for the positive class (`Revenue = True`). AUC uses the predicted probability of the positive class.

### Model comparison table

Values come from `results/model_comparison.csv` (test set, rounded to 4 decimals).

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8808 | 0.8828 | 0.7366 | 0.3586 | 0.4824 | 0.4592 |
| Decision Tree | 0.8585 | 0.7356 | 0.5420 | 0.5576 | 0.5497 | 0.4658 |
| KNN | 0.8767 | 0.7633 | 0.6822 | 0.3822 | 0.4899 | 0.4493 |
| Naive Bayes | 0.2729 | 0.7334 | 0.1726 | 0.9738 | 0.2933 | 0.1289 |
| Support Vector Machine | 0.8873 | 0.8824 | 0.7047 | 0.4686 | 0.5629 | 0.5149 |

### Observations on model performance

Because only about 15% of sessions end in a purchase, a model that always predicts "no purchase" would already score about 84-85% accuracy. Accuracy is therefore a weak guide here, and AUC, recall, F1 and MCC are used to judge the models.

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Highest AUC (0.8828) and the second-highest precision (0.7366), so it ranks buyers above non-buyers very well and is right most of the time when it predicts a purchase. Its recall is the lowest of all models (0.3586), so it misses nearly two thirds of real purchases at the default 0.5 threshold. |
| Decision Tree | Best recall among the four balanced models (0.5576) and a precision of 0.5420, so it finds more buyers but raises more false alarms. It has the lowest accuracy of that group (0.8585) and a weak AUC (0.7356), which suggests its unpruned splits fit the training data too closely and generalise less well. |
| KNN | Accuracy of 0.8767 and precision of 0.6822 look good, but recall is only 0.3822 and AUC is 0.7633. Neighbours in the majority "no purchase" class dominate the vote, so it under-detects buyers. MCC (0.4493) is the lowest of the four stronger models. |
| Naive Bayes | Highest recall (0.9738) but by far the lowest accuracy (0.2729) and precision (0.1726). It labels most sessions as purchases, which catches nearly every buyer but is useless in practice. The independence assumption does not hold here, since page counts and durations are strongly correlated. Its MCC (0.1289) shows it is barely better than guessing. |
| Support Vector Machine | Highest accuracy (0.8873), F1 (0.5629) and MCC (0.5149), with an AUC (0.8824) almost equal to Logistic Regression's. Its recall (0.4686) is better than Logistic Regression's and KNN's, while precision stays at 0.7047, so it gives the best balance between the two. |

**Overall winner for this dataset: Support Vector Machine.** It has the best F1 and MCC, which measure quality on the minority purchase class, and the best accuracy, while its AUC is within 0.0004 of the best (Logistic Regression). Logistic Regression is a close second and is faster and easier to interpret. The trade-off is that SVM trains and predicts more slowly, and its recall is still under 50%, so adjusting the decision threshold or using class weights could improve purchase detection further.

## Methodology

- **Split:** stratified 80:20 train/test split, so both sets keep the same share of purchases.
- **Numeric features:** median imputation, then standardisation.
- **Categorical features:** most-frequent imputation, then one-hot encoding (`handle_unknown="ignore"`).
- **Leakage control:** the split happens before any preprocessing is fitted.
- **Saved artefacts:** each model is saved as a single scikit-learn `Pipeline` (preprocessing plus classifier) in `model/`, so the app applies exactly the same preprocessing at prediction time.

## Streamlit app

The app (`app.py`) provides:

- CSV upload of a labelled test set (features and the `Revenue` column)
- A dropdown to choose one of the five saved models
- Evaluation metrics: Accuracy, AUC, Precision, Recall, F1, MCC
- Confusion matrix and classification report

The app loads the saved models and only evaluates them on the uploaded file. It **does not retrain** anything. Use `test_data.csv` from this repository as the upload.

## Run locally or in BITS Virtual Lab

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python train_models.py          # trains models, writes model/, results/, test_data.csv
streamlit run app.py
```

## Repository structure

```
shopper-purchase-intent-classifier/
├── app.py                        # Streamlit evaluation app
├── train_models.py               # preprocessing, training, evaluation, saving
├── requirements.txt              # dependencies
├── README.md                     # this file
├── test_data.csv                 # labelled hold-out test set for the app
├── online_shoppers_intention.csv # full source dataset
├── model/                        # five saved pipelines (.joblib)
└── results/
    └── model_comparison.csv      # metrics for all five models
```

## Deployment

Deployed on Streamlit Community Cloud from the `main` branch with `app.py` as the entry file.
