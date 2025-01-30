import os
import csv
import json
from tqdm import tqdm


# Directory containing the metrics files
exp_id = 'budget_scale'
exp_dir = os.path.join("experiments", exp_id)
exp_files = os.listdir(exp_dir)
rows = []

# Iterate over each file in the metrics directory
print(f"{len(exp_files)} files found")
for filename in tqdm(exp_files):
    if filename.endswith('.json'):
        file_path = os.path.join(exp_dir, filename)
        
        # Read the JSON file
        with open(file_path, 'r') as f:
            metrics = json.load(f)
        row = {'filename': filename.replace('.json', '')}
        
        scalar_keys = [k for k in metrics.keys() if not isinstance(metrics[k], (list, dict))]
        list_keys = [k for k in metrics.keys() if isinstance(metrics[k], list)]
        dict_keys = [k for k in metrics.keys() if isinstance(metrics[k], dict)] # baselines
        
        for k in scalar_keys:
            row[k] = metrics[k]
        for k in list_keys:
            row[f"{k}_first"] = metrics[k][0]
            row[f"{k}_last"] = metrics[k][-1]
        for k in dict_keys:
            for kk, vv in metrics[k].items():
                row[f"{k}_{kk}"] = vv
        rows.append(row)

# Write the rows to the CSV file
output_file = os.path.join("experiments", f'{exp_id}.csv')
with open(output_file, 'w', newline='') as csvfile:
    fieldnames = rows[0].keys() if rows else []
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)