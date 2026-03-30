# Exploratory Data Analysis

- The dataset is highly imbalanced, with fraudulent transactions representing only 0.17% of all data.
- Most features are PCA-transformed and already normalized, reducing the need for extensive preprocessing.
- The `Amount` feature is highly skewed and may require scaling or transformation.
- No missing values are present, simplifying data preparation.
- Due to class imbalance, evaluation metrics such as precision, recall, and F1-score will be more appropriate than accuracy.

## Implications

- Special techniques such as resampling (SMOTE) or anomaly detection will be necessary.
- Model performance should prioritize detecting fraud (recall) while minimizing false positives.

# Model Evaluation Insights

- The model achieves high recall, meaning it successfully detects most fraudulent transactions.
- Precision may be lower due to false positives, which is expected in fraud detection.
- ROC-AUC provides a better measure of overall performance than accuracy.

## Key Takeaway

In fraud detection, recall is often more important than precision, as missing fraudulent transactions can be more costly than flagging legitimate ones.
