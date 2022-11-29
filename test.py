import statistics as s

x = [1, 5, 7, 5, 43, 43, 8, 43, 6]

quartiles = s.quantiles(x, n=4)
print("Quartiles are: " + str(quartiles))
