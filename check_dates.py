import csv

f = open('data/uploaded/sampleb.csv', encoding='utf-8-sig')
reader = csv.DictReader(f)
row = next(reader)
print("Created Date sample:", repr(row.get('Created Date')))
print("Closed Date sample:", repr(row.get('Closed Date')))

# Try a few more samples
for i in range(3):
    row = next(reader)
    created = row.get('Created Date')
    closed = row.get('Closed Date')
    print(f"Row {i+2}: Created={repr(created)}, Closed={repr(closed)}")

f.close()
