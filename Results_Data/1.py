import pandas as pd


filepath = input("Enter file location:\t")
# epi_01/epitope_split_test_v1.csv
# tcr_01/tcr_split_test_v1.csv
data = pd.read_csv(filepath,delimiter='\t',header=None) 


print(data.head())
#remove 2nd and 3rd column
data2 = data.drop(data.columns[[2,3]], axis=1)

print(data2.head(5))

data2.to_csv('epi_01/epi_split_test.csv', sep='\t', index=False, header=False)



#command to generate requireemts.txt file
