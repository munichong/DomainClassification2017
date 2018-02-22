import csv

correct_set0999 = set()
with open('../Output/correct_predictions_0.999.csv') as infile:
    input = csv.reader(infile)
    next(input)
    for line in input:
        if not line:
            continue
        correct_set0999.add('.'.join(line[1]))

correct_set0001 = set()
with open('../Output/correct_predictions_0.001.csv') as infile:
    input = csv.reader(infile)
    next(input)
    for line in input:
        if not line:
            continue
        correct_set0001.add('.'.join(line[1]))

print(len(correct_set0999))
print(len(correct_set0001))
print(len(correct_set0999 & correct_set0001))
print(len(correct_set0999 | correct_set0001))