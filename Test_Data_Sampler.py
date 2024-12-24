import csv
import random

main_dataset_file = "data/Trnx_Dataset_v2.csv"

def sample_testing_data(user_file, sample_count):
    with open(main_dataset_file, newline='') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        rows = list(reader)

    random.shuffle(rows)
    sample_data = rows[:sample_count]

    # Remove columns "Reported by Authority", "Source of Money", "Money Laundering Risk Score" "Risk_Classification" and "Tax Haven Country"
    columns_to_remove = ["Reported by Authority", "Source of Money", "Money Laundering Risk Score", "Tax Haven Country", "Risk_Classification"]
    new_header = []
    for col in header:
        if col not in columns_to_remove:
            new_header.append(col)

    new_sample_data = []
    for row in sample_data:
        new_row = []
        for col, value in zip(header, row):
            if col not in columns_to_remove:
                new_row.append(value)
        new_sample_data.append(new_row)

    header = new_header
    sample_data = new_sample_data

    # Add a new column "Risk_Classification"
    header.append("Risk_Classification")

    with open(user_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(sample_data)

user_file = "Sample_User_Dataset.csv"
sample_testing_data(user_file, 3000)

