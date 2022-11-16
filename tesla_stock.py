import pandas as pd

#Tesla opening and closing stock return every day from Nov 16,2017 - Nov 16,2022
df = pd.read_csv("TSLA.csv",skiprows=0)
Date = df.iloc[:,0]
Open = df.iloc[:,1]
Close = df.iloc[:4]