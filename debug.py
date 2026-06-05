import pandas as pd

df = pd.read_csv('full.csv')
sway_cols = [c for c in df.columns if 'SWAY' in c]
print("SWAY columns found in CSV:")
print(sway_cols)