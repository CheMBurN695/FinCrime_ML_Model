import pandas as pd
from matplotlib import pyplot
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import numpy as npy

dataset_URL = "https://raw.githubusercontent.com/CheMBurN695/FinCrime_ML_Model/refs/heads/master/Trnx_Dataset.csv?token=GHSAT0AAAAAACYPUMUJT3IM6CAWCKWPPIDEZ3IRX6Q"
testing_dataset_URL = "https://raw.githubusercontent.com/CheMBurN695/FinCrime_ML_Model/refs/heads/master/Testing_Dataset.csv?token=GHSAT0AAAAAACYPUMUI6IF7VZEW3IITYWSWZ3ITRMA"
training_dataset_URL = "https://raw.githubusercontent.com/CheMBurN695/FinCrime_ML_Model/refs/heads/master/Training_Dataset.csv?token=GHSAT0AAAAAACYPUMUJW4YEFAPCRGSUTEZMZ3ITQ6Q"
dataFrame = pd.read_csv("data/Trnx_Dataset_v2.csv")
training_dataFrame = pd.read_csv("data/Training_Dataset.csv")
testing_dataFrame = pd.read_csv("data/Testing_Dataset.csv")

country_count = dataFrame.groupby(["Country"]).size()
illegal_trnxs = dataFrame['Source of Money'] == "Illegal"
illegal_trnx_byCountry = dataFrame[illegal_trnxs].groupby(["Country"]).size()
not_illegal_trnx_byCountry = country_count - illegal_trnx_byCountry
percent_illegal = (illegal_trnx_byCountry / country_count) * 100
percent_legal = (not_illegal_trnx_byCountry / country_count) * 100

_fig, _axes = pyplot.subplots(figsize=(8,6))
_axes.bar(illegal_trnx_byCountry.index, illegal_trnx_byCountry, color='red', label="Illegal")
_axes.bar(not_illegal_trnx_byCountry.index, not_illegal_trnx_byCountry, color='green', label="Legal")

_axes.set_title("Distribution of Illegal Source of Funds per Country In Dataset")
_axes.set_xlabel("Country")
_axes.set_ylabel("Transaction Count")
_axes.set_xticks(country_count.index)
_axes.set_xticklabels(country_count.index, rotation=45)
_axes.legend()
pyplot.show()

# Distribution of Risk scores
transaction_count  = dataFrame.groupby(["Money Laundering Risk Score"]).size()
transaction_count = transaction_count.sort_index()
transaction_count.plot(kind='bar', figsize=(8,6), color='blue')
pyplot.title("Distribution of Risk Scores")
pyplot.show()

Risk_score_count  = dataFrame.groupby(["Risk_Classification"]).size()
Risk_score_count = Risk_score_count.sort_index()
Risk_score_count.plot(kind='bar', figsize=(8,6), color='blue')
pyplot.title("Distribution of Risk Classifications")
pyplot.show()

transaction_types = dataFrame.groupby(["Transaction Type"]).size()
transaction_types.plot(kind = 'pie', autopct='%1.1f%%', figsize=(8,6))
pyplot.title("Distribution of Transaction Types")
pyplot.show()

label_encoder_country = LabelEncoder()
label_encoder_destination_country = LabelEncoder()
label_encoder_transaction_type = LabelEncoder()
label_encoder_person_involved = LabelEncoder()
label_encoder_industry = LabelEncoder()
label_encoder_source_of_money = LabelEncoder()
label_encoder_ML_score = LabelEncoder()
label_encoder_ML_class = LabelEncoder()
label_encoder_shellCosInvolved = LabelEncoder()

label_encoder_country.fit(dataFrame['Country'])
label_encoder_destination_country.fit(dataFrame['Destination Country'])
label_encoder_transaction_type.fit(dataFrame['Transaction Type'])
label_encoder_person_involved.fit(dataFrame['Person Involved'])
label_encoder_industry.fit(dataFrame['Industry'])
label_encoder_source_of_money.fit(dataFrame['Source of Money'])
label_encoder_ML_score.fit(dataFrame['Money Laundering Risk Score'])
label_encoder_ML_class.fit(dataFrame['Risk_Classification'])
label_encoder_shellCosInvolved.fit(dataFrame['Shell Companies Involved'])

encoded_df = dataFrame.copy()
encoded_df['Country'] = label_encoder_country.transform(encoded_df['Country'])
encoded_df['Destination Country'] = label_encoder_destination_country.transform(encoded_df['Destination Country'])
encoded_df['Transaction Type'] = label_encoder_transaction_type.transform(encoded_df['Transaction Type'])
encoded_df['Person Involved'] = label_encoder_person_involved.transform(encoded_df['Person Involved'])
encoded_df['Industry'] = label_encoder_industry.transform(encoded_df['Industry'])
encoded_df['Source of Money'] = label_encoder_source_of_money.transform(encoded_df['Source of Money'])
encoded_df['Money Laundering Risk Score'] = label_encoder_ML_score.transform(encoded_df['Money Laundering Risk Score'])
encoded_df['Risk_Classification'] = label_encoder_ML_class.transform(encoded_df['Risk_Classification'])
encoded_df['Shell Companies Involved'] = label_encoder_shellCosInvolved.transform(encoded_df['Shell Companies Involved'])

for columns in encoded_df.columns:
    if encoded_df[columns].dtype == 'object':
        encoded_df.drop(columns, axis=1, inplace=True)

# Correlation matrix
corr_matrix = encoded_df.corr()
pyplot.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='Blues')
pyplot.title("Correlation Matrix")
pyplot.show()

#Encode training data

training_dataFrame['Country'] = label_encoder_country.transform(training_dataFrame['Country'])
training_dataFrame['Destination Country'] = label_encoder_destination_country.transform(training_dataFrame['Destination Country'])
training_dataFrame['Transaction Type'] = label_encoder_transaction_type.transform(training_dataFrame['Transaction Type'])
training_dataFrame['Person Involved'] = label_encoder_person_involved.transform(training_dataFrame['Person Involved'])
training_dataFrame['Industry'] = label_encoder_industry.transform(training_dataFrame['Industry'])
training_dataFrame['Source of Money'] = label_encoder_source_of_money.transform(training_dataFrame['Source of Money'])
training_dataFrame['Money Laundering Risk Score'] = label_encoder_ML_score.transform(training_dataFrame['Money Laundering Risk Score'])
training_dataFrame['Risk_Classification'] = label_encoder_ML_class.transform(training_dataFrame['Risk_Classification'])
training_dataFrame['Shell Companies Involved'] = label_encoder_shellCosInvolved.transform(training_dataFrame['Shell Companies Involved'])

# X = training_dataFrame[['Country', 'Destination Country', 'Transaction Type', 'Person Involved', 'Industry', 'Source of Money', 'Shell Companies Involved']]
# X = training_dataFrame[['Country', 'Destination Country', 'Person Involved', 'Industry', 'Source of Money', 'Shell Companies Involved']]
# X = training_dataFrame[['Country', 'Destination Country', 'Industry', 'Source of Money', 'Shell Companies Involved', 'Transaction Type']]
X = training_dataFrame[['Country', 'Destination Country', 'Industry', 'Shell Companies Involved', 'Transaction Type']]
Y = training_dataFrame['Risk_Classification']

# X_grad = training_dataFrame[['Country', 'Destination Country', 'Person Involved', 'Industry', 'Source of Money', 'Shell Companies Involved']]
X_grad = training_dataFrame[['Country', 'Destination Country', 'Industry', 'Source of Money', 'Shell Companies Involved', 'Transaction Type']]
Y_grad = training_dataFrame['Money Laundering Risk Score']


# testing data

testing_dataFrame['Country'] = label_encoder_country.transform(testing_dataFrame['Country'])
testing_dataFrame['Destination Country'] = label_encoder_destination_country.transform(testing_dataFrame['Destination Country'])
testing_dataFrame['Transaction Type'] = label_encoder_transaction_type.transform(testing_dataFrame['Transaction Type'])
testing_dataFrame['Person Involved'] = label_encoder_person_involved.transform(testing_dataFrame['Person Involved'])
testing_dataFrame['Industry'] = label_encoder_industry.transform(testing_dataFrame['Industry'])
testing_dataFrame['Source of Money'] = label_encoder_source_of_money.transform(testing_dataFrame['Source of Money'])
testing_dataFrame['Money Laundering Risk Score'] = label_encoder_ML_score.transform(testing_dataFrame['Money Laundering Risk Score'])
testing_dataFrame['Risk_Classification'] = label_encoder_ML_class.transform(testing_dataFrame['Risk_Classification'])
testing_dataFrame['Shell Companies Involved'] = label_encoder_shellCosInvolved.transform(testing_dataFrame['Shell Companies Involved'])

# X_test = testing_dataFrame[['Country', 'Destination Country', 'Person Involved', 'Industry', 'Source of Money', 'Shell Companies Involved']]
# X_test = testing_dataFrame[['Country', 'Destination Country', 'Industry', 'Source of Money', 'Shell Companies Involved', 'Transaction Type']]
X_test = testing_dataFrame[['Country', 'Destination Country', 'Industry', 'Shell Companies Involved', 'Transaction Type']]
Y_test = testing_dataFrame['Risk_Classification']

# X_grad_testing = training_dataFrame[['Country', 'Destination Country', 'Person Involved', 'Industry', 'Source of Money', 'Shell Companies Involved']]
X_grad_testing = training_dataFrame[['Country', 'Destination Country', 'Industry', 'Source of Money', 'Shell Companies Involved', 'Transaction Type']]
Y_grad_testing = training_dataFrame['Money Laundering Risk Score']

model = LogisticRegression(max_iter=1000)
model.fit(X,Y)
predictions = model.predict(X_test)


conf_matrix = confusion_matrix(Y_test, predictions)
pyplot.figure(figsize=(10, 8))

conf_matrix_updated = npy.array([
    [conf_matrix[0, 0], conf_matrix[0, 1]],
    [conf_matrix[1, 0], conf_matrix[1, 1]]
])

# Define the new labels
labels = ["Low Risk", "High Risk"]
sns.heatmap(conf_matrix_updated, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
pyplot.xlabel('Predicted Labels')
pyplot.ylabel('True Labels')
pyplot.title('Confusion Matrix')
pyplot.show()

class_report = classification_report(Y_test, predictions, target_names=labels)
print(class_report)


rf_model = RandomForestClassifier(n_estimators=300, random_state=420)
rf_model.fit(X, Y)
rf_predictions = rf_model.predict(X_test)

print("Accuracy:", accuracy_score(Y_test, rf_predictions))
print("Classification Report:\n", classification_report(Y_test, rf_predictions))

conf_matrix = confusion_matrix(Y_test, rf_predictions)
pyplot.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')

pyplot.xlabel('Predicted Labels')
pyplot.ylabel('True Labels')
pyplot.title('Confusion Matrix')
pyplot.show()

import ipywidgets as widgets
from IPython.display import display, clear_output

country_options = ['USA', 'South Africa', 'Switzerland', 'Russia', 'Brazil', 'Singapore', 'UK', 'UAE', 'China']
transaction_type_options = ['Offshore Transfer', 'Stocks Transfer', 'Cryptocurrency', 'Property Purchase', 'Cash Withdrawal']
industry_options = ['Finance', 'Casinos', 'Construction', 'Luxury Goods', 'Oil & Gas', 'Real Estate', 'Arms Trade']

country_dropdown = widgets.Dropdown(options=country_options, description="Country:")
destination_country_dropdown = widgets.Dropdown(options=country_options, description="Dest. Country:")
industry_dropdown = widgets.Dropdown(options=industry_options, description="Industry:")
shell_companies_input = widgets.IntText(description="Shell Cos:")
transaction_type_dropdown = widgets.Dropdown(options=transaction_type_options, description="Trnx Type:")

predict_button = widgets.Button(description="Predict Risk")
output = widgets.Output()

def predict_risk(button):
    with output:
        clear_output()

        input_data = {
            "Country": [country_dropdown.value],
            "Destination Country": [destination_country_dropdown.value],
            "Industry": [industry_dropdown.value],
            "Shell Companies Involved": [shell_companies_input.value],
            "Transaction Type": [transaction_type_dropdown.value]
        }

        input_df = pd.DataFrame(input_data)

        # use same encoder
        input_df['Country'] = label_encoder_country.transform(input_df['Country'])
        input_df['Destination Country'] = label_encoder_destination_country.transform(input_df['Destination Country'])
        input_df['Industry'] = label_encoder_industry.transform(input_df['Industry'])
        input_df['Transaction Type'] = label_encoder_transaction_type.transform(input_df['Transaction Type'])

        prediction = rf_model.predict(input_df)

        risk_class = "High Risk" if prediction[0] == 1 else "Low Risk"
        print(f"Predicted Risk Classification: {risk_class}")

predict_button.on_click(predict_risk)
display(country_dropdown, destination_country_dropdown, industry_dropdown, shell_companies_input, transaction_type_dropdown, predict_button, output)