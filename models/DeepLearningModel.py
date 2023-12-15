import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Flatten
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np

# dataset, pre-processed and loaded here (after Adam's update)
X = ...  # input features; 3D for CNN (samples, timesteps, features)
y = ...  # output labels

# one hot encode the the output labels
def one_hot_encode_seq(sequence, max_length):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}

    # Pad sequence with 'N' if it's shorter than max_length
    padded_sequence = sequence + 'N' * (max_length - len(sequence))

    seq_encoded = [mapping[nuc] for nuc in padded_sequence]
    one_hot_matrix = np.eye(5) 
    return one_hot_matrix[seq_encoded]


# Split the data into training and testing sets -- I believe that to_categorical changes the labels to one-hot encoded vectors
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_train, y_test = to_categorical(y_train), to_categorical(y_test)  # classification labels

# building the model (CNN + LSTM)
model = Sequential()

# CNN layer -- make sure the input shape matches the shape of the input data
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', strides=1, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(MaxPooling1D(pool_size=2))

# LSTM layer
model.add(LSTM(50, return_sequences=True))
model.add(Flatten())

# Dense layer for output
model.add(Dense(100, activation='relu'))
model.add(Dense(y_train.shape[1], activation='softmax'))  # Will need to adjust # of neurons to match # of output classes

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Model summary
model.summary()

# Train the model -- maybe change epochs and batch_size lateradam
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy}")

