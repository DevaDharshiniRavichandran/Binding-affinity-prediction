import sys
import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from keras.optimizers import Adam

# Helper function for one-hot encoding
def one_hot_encode(sequence, max_length, vocab):
    encoding = np.zeros((max_length, len(vocab)))
    for i, char in enumerate(sequence):
        if i < max_length and char in vocab:
            encoding[i, vocab.index(char)] = 1
    return encoding

# Function to preprocess test data
def preprocess_test_data(test_file, max_epitope_length=15, max_tcr_length=30):
    tcr_data = pd.read_csv(test_file, header=None)

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

    # Flatten embeddings
    epi_embeddings_flat = epi_embeddings.reshape(epi_embeddings.shape[0], -1)
    tcr_embeddings_flat = tcr_embeddings.reshape(tcr_embeddings.shape[0], -1)

    return epi_embeddings_flat, tcr_embeddings_flat, binding

# Function to evaluate the model
def test_model(test_file):
    model = load_model('epi01_model.h5', compile=False)
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])

    # Preprocess test data
    epi_embeddings, tcr_embeddings, binding = preprocess_test_data(test_file)

    # Make predictions
    yhat = model.predict([epi_embeddings, tcr_embeddings])

    # Calculate performance metrics
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
    with open('EPI01_test_performance.txt', 'w') as f:
        f.write('Performance on Test Data:\n')
        f.write(f'AUC: {auc}\n')
        f.write(f'Accuracy: {accuracy}\n')
        f.write(f'Precision: {precision}\n')
        f.write(f'Recall: {recall}\n')
        f.write(f'F1 Score: {f1}\n')

def main():
    if len(sys.argv) != 2:
        print("Usage: python test_model.py epi_test.csv")
        sys.exit(1)

    test_file = sys.argv[1]
    test_model(test_file)

if __name__ == "__main__":
    main()
