import pandas as pd
import re
import time
import os
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
path = '../data/df_data'
training_amount = 1


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


def train(train_files):
    print('  Training Naive Bays on ' + str(len(train_files)) + ' file/s:\n    ' + str(train_files))
    dfs_train = [pd.read_csv(file) for file in train_files]
    df_train = pd.concat(dfs_train, ignore_index=True)
    X_train = df_train.iloc[:, :-1]  # Features
    y_train = df_train.iloc[:, -1]   # Target variable
    model = GaussianNB()
    print('  File Aggregation Complete\n  Training Model')
    model.fit(X_train, y_train)
    print('  Training Complete')
    return model


def evaluate(model, test_files):
    print('  Testing Model on ' + str(len(test_files)) + ' file/s:\n    ' + str(test_files))
    dfs_test = [pd.read_csv(file) for file in test_files]
    df_test = pd.concat(dfs_test, ignore_index=True)
    X_test = df_test.iloc[:, :-1]  # Features
    y_test = df_test.iloc[:, -1]  # Target variable
    y_pred = model.predict(X_test)
    # Calculate and print accuracy, precision, recall, and F1 score
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred, average='weighted')
    recall = metrics.recall_score(y_test, y_pred, average='weighted')
    f1 = metrics.f1_score(y_test, y_pred, average='weighted')
    print('  Testing Complete:')
    print(f'    - Accuracy: {100 * round(accuracy, 3)}%')
    print(f'    - Precision: {100 * round(precision, 3)}%')
    print(f'    - Recall: {100 * round(recall, 3)}%')
    print(f'    - F1 Score: {100 * round(f1, 3)}%')


def main():
    start = time.time()
    print('Naive Bays Algorithm')
    files = os.listdir(path)
    files = sorted(files, key=natural_sort_key)
    train_files = []
    test_files = []
    for elem in files:
        if files.index(elem) < training_amount:
            train_files.append(path + '/' + elem)
        else:
            test_files.append(path + '/' + elem)
    model = train(train_files)
    evaluate(model, test_files)
    end = time.time()
    print("Completed in: " + str(round(end-start, 4)) + " seconds")


if __name__ == '__main__':
       main()
