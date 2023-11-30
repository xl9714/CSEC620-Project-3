import pandas as pd
import numpy as np
import math
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class CW(object):

    def __init__(self, n_features, PSI = 0.95, initial_mean_value = 0, initial_variance_value = 1):
        self.n_features = n_features
        self.p = PSI
        self.m = np.zeros(n_features) + initial_mean_value
        self.S = np.zeros(n_features) + initial_variance_value
          
    def update(self, x, y):
        m_t = np.dot(self.m.T, x)
        v_t = np.dot(np.dot(x.T, np.diag(self.S)), x)
        alpha_t = max(0, (- m_t * y * self.p + math.sqrt((m_t * y)**2 * self.p**2 + 4 * v_t * self.p)) / (2 * v_t))
        u_t = - m_t * y + alpha_t * v_t
        phi_t = self._phi(-u_t)
        z_t = phi_t/(phi_t - self.p) * u_t
        self.m = self.m + alpha_t * y * self.S * x
        gamma_t = 1 + 2 * alpha_t * phi_t
        delta_t = z_t + alpha_t * phi_t
        eta_t = 1 / gamma_t * (1 - delta_t)
        self.S = self.S * eta_t

    def predict(self, x):
        if np.dot(x, self.m) > 0:
            return 1
        else:
            return -1

    def _phi(self, x):
        return 1/2/np.pi*np.exp(-x**2/2)

# Load training data
train_data = pd.read_csv('newfile0.csv')

# Split the dataset into features and target variable
X_train = train_data.iloc[:,:-1].values
y_train = train_data.iloc[:,-1].values

# Initialize the classifier
clf = CW(X_train.shape[1])

# Train the classifier
for i in range(X_train.shape[0]):
    clf.update(X_train[i], y_train[i])

total_accuracy = 0 
total_precision = 0 
total_recall = 0 
total_f1 = 0 
num_files = 119

# Test the classifier with the other files and calculate scores
for j in range(1, num_files):
    test_data = pd.read_csv(f"newfile{j}.csv")
    X_test = test_data.iloc[:,:-1].values
    y_test = test_data.iloc[:,-1].values
    y_pred = [clf.predict(x) for x in X_test]
  
    total_accuracy += accuracy_score(y_test, y_pred)
    total_precision += precision_score(y_test, y_pred, average='weighted')
    total_recall += recall_score(y_test, y_pred, average='weighted')
    total_f1 += f1_score(y_test, y_pred, average='weighted')

# Print the average metrics over all test files
print(f"Average Accuracy: {total_accuracy / num_files}")
print(f"Average Precision: {total_precision / num_files}")
print(f"Average Recall: {total_recall / num_files}")
print(f"Average F1 Score: {total_f1 / num_files}")