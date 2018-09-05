import csv
import numpy as np

def read(file, data):
    record = []
    result = np.zeros(3)
    total = 0
    with open(file, 'r', encoding = 'utf-8') as csvFile:
        csvReader = csv.reader(csvFile)
        for i in range(4):
            next(csvReader)
        for row in csvReader:
            print(row[3:5])
            record.append(list(map(float, row[3:5])))
        for line in record:
            total += line[1] - line[0]
            result = np.vstack((result, data[int(line[0]):int(line[1])]))
    return np.array(result)

surrogate = np.random.rand(300000, 3)
result = read('/Users/preston/SCH-Local/Preston Wearing Log.csv', surrogate)
