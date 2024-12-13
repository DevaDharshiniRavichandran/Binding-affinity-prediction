import os
import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Function to load and preprocess data
def one_hot_encode(sequence, max_length, vocab):
    encoding = np.zeros((max_length, len(vocab)))
    for i, char in enumerate(sequence):
        if i < max_length and char in vocab:
            encoding[i, vocab.index(char)] = 1
    return encoding

# Function to preprocess data
def load_test_data(tcr_file, max_epitope_length=15, max_tcr_length=30):
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

# Function to test the model
def test_model(tcr_test_file, model_file):
    # Load test data
    epi_embeddings, tcr_embeddings, binding = load_test_data(tcr_test_file)

    epi_embeddings = epi_embeddings.reshape(epi_embeddings.shape[0], -1)
    tcr_embeddings = tcr_embeddings.reshape(tcr_embeddings.shape[0], -1)

    # Load the trained model
    model = load_model(model_file)

    # Evaluate the model on the test set
    yhat = model.predict([epi_embeddings, tcr_embeddings])
    print('================Performance on Test Data===================')
    auc = roc_auc_score(binding, yhat)
    yhat_binary = (yhat >= 0.5).astype(int)
    accuracy = accuracy_score(binding, yhat_binary)
    precision = precision_score(binding, yhat_binary)
    recall = recall_score(binding, yhat_binary)
    f1 = f1_score(binding, yhat_binary)

    # Print metrics to console
    print(f'AUC: {auc}')
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')

    # Save performance metrics to a file
    with open('EPI_test_performance.txt', 'w') as f:
        f.write('Performance on Test Data:\n')
        f.write(f'AUC: {auc}\n')
        f.write(f'Accuracy: {accuracy}\n')
        f.write(f'Precision: {precision}\n')
        f.write(f'Recall: {recall}\n')
        f.write(f'F1 Score: {f1}\n')

# Main function
def main():
    tcr_test_file = 'epi_test.csv'
    model_file = 'epi_model.h5'

    test_model(tcr_test_file, model_file)

if __name__ == '__main__':
    main()
