import pandas as pd
import json

data = pd.read_json('data.json')

print(data.head())