import csv
import random

testing_dataset = "Testing_Dataset.csv"
training_dataset = "Training_Dataset.csv"

def split_training_testing_samples(_dataset_file):
    with open(_dataset_file, newline='') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        rows = list(reader)

    random.shuffle(rows)
    split_index = int(0.8 * len(rows))
    training_data = rows[:split_index]
    testing_data = rows[split_index:]

    with open(training_dataset, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(training_data)

    with open(testing_dataset, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(testing_data)

datasetFile = "Trnx_Dataset.csv"
# split_training_testing_samples(datasetFile)

# Data cleaner

hugeDataset = "G:\\My Drive\\College Stuff\\Capstone\\HI-Small_Trans.csv"
outputCSV = "IBM_Dataset.csv"

def work_the_data():
    with open(hugeDataset, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        header = reader.fieldnames

        laundering_rows = []
        other_rows = []

        for row in reader:
            if row['Is Laundering'] == '1':
                laundering_rows.append(row)
            else:
                other_rows.append(row)

        random.shuffle(other_rows)

        total_required = 700000 - len(laundering_rows)
        additional_rows = other_rows[:total_required]
        all_rows = laundering_rows + additional_rows

        with open(outputCSV, 'w', newline='') as output_file:
            writer = csv.DictWriter(output_file, fieldnames=header)
            writer.writeheader()
            writer.writerows(all_rows)

work_the_data()