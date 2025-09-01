import os
import json
import csv

input_folder = "data"
output_file = "pose_dataset.csv"

all_rows = []
header = [f"{axis}{i}" for i in range(33) for axis in ['x', 'y', 'z', 'visibility']] + ['label']

for file in os.listdir(input_folder):
    if file.endswith(".json"):
        with open(os.path.join(input_folder, file), 'r') as f:
            data = json.load(f)
            row = []
            for lm in data['landmarks']:
                row.extend([lm['x'], lm['y'], lm['z'], lm['visibility']])
            row.append(data['label'])
            all_rows.append(row)

# Write to CSV
with open(output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(all_rows)

print(f"✅ Saved CSV with {len(all_rows)} samples to {output_file}")
