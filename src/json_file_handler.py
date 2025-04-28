import json
import sys
import os

# Open and read the JSON file
file_path = os.path.join('..', 'data', 'iris_data_fix.json')
with open(file_path, 'r') as f:
    data = json.load(f)

# Now 'data' is a Python dictionary (or list, depending on the JSON structure)
print(data)
