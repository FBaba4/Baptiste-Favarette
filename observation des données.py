import pandas as pd

file_path='/home/favarette/MAF/donnees_structurees.csv'
df=pd.read_csv(file_path)
df.describe()
print (df.head(10))