#test if csv is being made
import os
import csv
from datetime import datetime

csv_folder = "csv"
if not os.path.exists(csv_folder):
    os.makedirs(csv_folder)

csv_file = os.path.join(csv_folder, f"test_file_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv")

header = ["column1", "column2", "column3"]
rows = [["data1", "data2", "data3"], ["data4", "data5", "data6"]]

if not os.path.exists(csv_file):
    with open(csv_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

print(f"CSV file created at: {csv_file}")
