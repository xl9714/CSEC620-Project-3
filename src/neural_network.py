import os
import glob

import pandas as pd
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from keras import backend as K
import time


path = "../data/df_data"
all_files = glob.glob(os.path.join(path + "/*.csv"))

labels = []
features = []

training_data = pd.read_csv(all_files[0], delimiter=",", header=0)
train_labels = training_data['target']
train_features = training_data.drop('target', axis=1)

for file_ind in range(1, len(all_files)):
    file = all_files[file_ind]
    dataset = pd.read_csv(file, delimiter=",", header=0)
    label = dataset['target']
    feature = dataset.drop('target', axis=1)
    labels.append(label)
    features.append(feature)

test_labels = pd.concat(labels, ignore_index=True)
test_features = pd.concat(features, ignore_index=True)

test_labels = test_labels.sample(frac=1, random_state=42)
test_features = test_features.sample(frac=1, random_state=42)

scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)

train_labels = to_categorical(train_labels, num_classes=3)
test_labels = to_categorical(test_labels, num_classes=3)

model = Sequential()
model.add(Dense(128, input_dim=train_features.shape[1], activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(3, activation='sigmoid'))

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

start_time = time.time()
# compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['acc',f1_m,precision_m, recall_m])


early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True)

# fit the model
history = model.fit(train_features, train_labels, epochs=100, batch_size=100, validation_split=0.1, verbose=0, callbacks=[early_stopping, model_checkpoint])

# evaluate the model
loss, accuracy, f1_score, precision, recall = model.evaluate(test_features, test_labels, verbose=0)

end_time = time.time()
evaluation_time = end_time - start_time

print("Test loss: ", loss)
print("Test accuracy: ", accuracy)
print("F1_score: ", f1_score)
print("Precision: ", precision)
print("Recall: ", recall)
print("Time spent: ", evaluation_time)