import pandas as pd
import csv

#Tesla opening and closing stock return every day from Nov 16,2017 - Nov 16,2022
df = pd.read_csv("TSLA.csv",skiprows=0)
Date = df.iloc[:,0]
Open = df.iloc[:,1]
Close = df.iloc[:4]
print(Date)

csv_output = "tesla_stock_return_Test.csv"
fields = ['Date','Average Stock Return']

with open(csv_output, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    
    #Write Fields to CSV
    csvwriter.writerow(fields)

    #Write Daily Average Stock Return
    for i in range(len(Date)):
        csvwriter.writerow([Date[i],(float(Open[i])+float(Open[i]))/2])

