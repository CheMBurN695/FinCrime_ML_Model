This project demonstrates the use of Machine Learning to classify financial transactions as High Risk or Low Risk, with the goal of helping detect suspicious activities and combat money laundering. Using a synthetic dataset of 10,000 financial transactions from Kaggle, the focus is on analyzing key features like country, destination country, industry, shell companies, and transaction type. I implemented Logistic Regression and Random Forest Classifier to tackle the classification problem.

Logistic Regression was chosen as the baseline due to its simplicity for binary classification. Sadly, it only achieved 66% accuracy and struggled with high-risk cases, as it had low recall for this class. The model often classified transactions as low-risk, which led to a performance imbalance. To improve, we switched to the Random Forest Classifier, which uses an ensemble approach. With 300 estimators and a random seed for consistency, this model performed much better, reaching an accuracy of 79.15%. It also showed a more balanced performance across both risk classes, with macro average precision, recall, and F1-score of 0.77, which exceeded our project goals.

To make this more accessible, I created an interactive user interface in Jupyter Notebook using ipywidgets, allowing users to input transaction details and get real-time risk predictions.

Random Forests

Accuracy: 0.7915
Classification Report:
               precision    recall  f1-score   support

           0       0.85      0.83      0.84      1320
           1       0.69      0.71      0.70       680

    accuracy                           0.79      2000
   macro avg       0.77      0.77      0.77      2000
weighted avg       0.79      0.79      0.79      2000


Logistic Regression

              precision    recall  f1-score   support

    Low Risk       0.68      0.91      0.78      1320
   High Risk       0.52      0.19      0.27       680

    accuracy                           0.66      2000
   macro avg       0.60      0.55      0.53      2000
weighted avg       0.63      0.66      0.61      2000
