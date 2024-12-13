import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Dropout, BatchNormalization, concatenate, Activation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from keras.callbacks import EarlyStopping, CSVLogger
from keras.optimizers import Adam


# Helper function for one-hot encoding
def one_hot_encode(sequence, max_length, vocab):
    encoding = np.zeros((max_length, len(vocab)))
    for i, char in enumerate(sequence):
        if i < max_length and char in vocab:
            encoding[i, vocab.index(char)] = 1
    return encoding

# Function to preprocess data
def load_data(tcr_file, max_epitope_length=15, max_tcr_length=30):
    tcr_data = pd.read_csv(tcr_file, header=None)

    # Vocabulary for one-hot encoding (all unique characters in sequences)
    vocab = sorted(set("".join(tcr_data[0]) + "".join(tcr_data[1])))

    # Encode epitope sequences
    epi_sequences = tcr_data.iloc[:, 0].dropna()
    epi_embeddings = np.array([one_hot_encode(seq, max_epitope_length, vocab) for seq in epi_sequences])

    # Encode TCR sequences
    tcr_sequences = tcr_data.iloc[:, 1].dropna()
    tcr_embeddings = np.array([one_hot_encode(seq, max_tcr_length, vocab) for seq in tcr_sequences])

    # Process binding labels
    binding = tcr_data.iloc[:, 2].dropna().values.astype(int)

    # Ensure all arrays have the same length
    min_len = min(len(epi_embeddings), len(tcr_embeddings), len(binding))
    epi_embeddings = epi_embeddings[:min_len]
    tcr_embeddings = tcr_embeddings[:min_len]
    binding = binding[:min_len]

    return epi_embeddings, tcr_embeddings, binding

# Function to train the model
def train_model(epi_embeddings, tcr_embeddings, binding):
    # Flatten the one-hot encoded arrays for input to the dense layers
    epi_embeddings = epi_embeddings.reshape(epi_embeddings.shape[0], -1)
    tcr_embeddings = tcr_embeddings.reshape(tcr_embeddings.shape[0], -1)

    # Split the data
    train_size = int(0.8 * len(binding))
    X1_train, X2_train, y_train = epi_embeddings[:train_size], tcr_embeddings[:train_size], binding[:train_size]
    X1_test, X2_test, y_test = epi_embeddings[train_size:], tcr_embeddings[train_size:], binding[train_size:]

    # Define the model
    inputA = Input(shape=(X1_train.shape[1],))
    inputB = Input(shape=(X2_train.shape[1],))

    x = Dense(2048, kernel_initializer='he_uniform')(inputA)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Activation('relu')(x)
    
    y = Dense(2048, kernel_initializer='he_uniform')(inputB)
    y = BatchNormalization()(y)
    y = Dropout(0.3)(y)
    y = Activation('relu')(y)

    combined = concatenate([x, y])
    z = Dense(1024)(combined)
    z = BatchNormalization()(z)
    z = Dropout(0.3)(z)
    z = Activation('relu')(z)
    z = Dense(1, activation='sigmoid')(z)

    model = Model(inputs=[inputA, inputB], outputs=z)
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])
    model.summary()

    # Callbacks: EarlyStopping and CSVLogger to save training history
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30)
    csv_logger = CSVLogger('EPI01_training_history.csv', append=False)

    # Train the model
    history = model.fit([X1_train, X2_train], y_train, 
                        validation_split=0.2, 
                        epochs=200, 
                        batch_size=32, 
                        callbacks=[es, csv_logger], 
                        verbose=1)

    # Save the trained model
    model.save('epi01_model.h5')

    # Evaluate the model on the test set
    yhat = model.predict([X1_test, X2_test])
    print('================Performance on Validation Data===================')
    auc = roc_auc_score(y_test, yhat)
    yhat_binary = (yhat >= 0.5).astype(int)
    accuracy = accuracy_score(y_test, yhat_binary)
    precision = precision_score(y_test, yhat_binary)
    recall = recall_score(y_test, yhat_binary)
    f1 = f1_score(y_test, yhat_binary)

    # Print metrics to console
    print(f'AUC: {auc}')
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')

    # Save performance metrics to a file
    with open('EPI01_validation_performance.txt', 'w') as f:
        f.write('Performance on Validation Data:\n')
        f.write(f'AUC: {auc}\n')
        f.write(f'Accuracy: {accuracy}\n')
        f.write(f'Precision: {precision}\n')
        f.write(f'Recall: {recall}\n')
        f.write(f'F1 Score: {f1}\n')

# Main function
def main():
    tcr_file = 'epi_train.csv'

    epi_embeddings, tcr_embeddings, binding = load_data(tcr_file)
    train_model(epi_embeddings, tcr_embeddings, binding)

if __name__ == '__main__':
    main()
