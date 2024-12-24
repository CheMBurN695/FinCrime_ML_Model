import csv
import random

testing_dataset = "data/Testing_Dataset.csv"
training_dataset = "data/Training_Dataset.csv"

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

datasetFile = "data/Trnx_Dataset_v2.csv"
split_training_testing_samples(datasetFile)

