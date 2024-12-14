# import pandas as pd
# import numpy as np
# # import random

# data = pd.read_csv('data/epitope_split_test.csv')
# data[len(data.columns)] =  np.random.choice([0, 1], size=len(data))

# data.to_csv('./tcr_split_test.csv', index=False)


# data = pd.read_csv('./tcr_split_test.csv')

# print(data.head(5))


import csv

# Input and output file paths
input_file = "result/tcr_ep20_0.01/pred_modified_tcr_split_test.csv"  # Replace with your actual file path
output_file = "result/pred_modified_tcr_split_test.csv"  # Name of the new file without the third column

# Read and process the file
with open(input_file, "r") as infile, open(output_file, "w", newline="") as outfile:
    reader = csv.reader(infile, delimiter="\t")  # Assuming tab-delimited CSV
    writer = csv.writer(outfile, delimiter="\t")

    for row in reader:
        # Remove the third column (index 2) if it exists
        if len(row) > 2:
            row.pop(2)
        writer.writerow(row)

print(f"Processed file saved as {output_file}")
