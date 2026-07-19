import pandas as pd

# read data
dt = pd.read_csv("data/raw/train.csv")

# randomly sample 150 rows
dt_sample = dt.sample(n=150, random_state=42)

dt_sample.to_csv("tests/sample_data/train_sample.csv")


