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

for file in all_files:
    dataset = pd.read_csv(file, delimiter=",", header=0)
    label = dataset['target']
    feature = dataset.drop('target', axis=1)
    labels.append(label)
    features.append(feature)

df_labels = pd.concat(labels, ignore_index=True)
df_features = pd.concat(features, ignore_index=True)

df_labels = df_labels.sample(frac=1, random_state=42)
df_features = df_features.sample(frac=1, random_state=42)

# df_labels = df_labels.fillna(df_labels.median())

# quantile_95 = dataset.quantile(0.95)
# dataset = np.where(dataset > quantile_95, quantile_95, dataset)

data_entry_size = df_features.shape[0]
training_size = int(data_entry_size * 0.8)
testing_size = int(data_entry_size * 0.2)

train_features = df_features[:training_size]
train_labels = df_labels[:training_size]

test_features = df_features[training_size:]
test_labels = df_labels[training_size:]

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